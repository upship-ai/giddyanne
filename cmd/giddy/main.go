// Package main provides a CLI client for giddyanne semantic code search.
package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	pidDir       = ".giddyanne"
	pollInterval = 500 * time.Millisecond
	defaultHost  = "0.0.0.0"
	defaultPort  = 8000
)

// SearchRequest is the API request for /search.
type SearchRequest struct {
	Query      string `json:"query"`
	Limit      int    `json:"limit"`
	SearchType string `json:"search_type"`
}

// SearchResult represents a single search result from the API.
type SearchResult struct {
	Path      string  `json:"path"`
	StartLine int     `json:"start_line"`
	EndLine   int     `json:"end_line"`
	Content   string  `json:"content"`
	Score     float64 `json:"score"`
}

// SearchResponse is the API response for /search.
type SearchResponse struct {
	Query   string         `json:"query"`
	Results []SearchResult `json:"results"`
}

// StatusResponse is the API response for /status.
type StatusResponse struct {
	State   string `json:"state"`
	Total   int    `json:"total"`
	Indexed int    `json:"indexed"`
	Percent int    `json:"percent"`
	Error   string `json:"error,omitempty"`
}

// StatsResponse is the API response for /stats.
type StatsResponse struct {
	IndexedFiles      int            `json:"indexed_files"`
	TotalChunks       int            `json:"total_chunks"`
	IndexSizeBytes    int64          `json:"index_size_bytes"`
	AvgQueryLatencyMs *float64       `json:"avg_query_latency_ms"`
	StartupDurationMs *float64       `json:"startup_duration_ms"`
	Files             map[string]int `json:"files"` // path -> chunk count
}

// commands defines all available commands with their full names.
var commands = []string{"up", "down", "bounce", "find", "log", "status", "health", "init", "mcp", "clean", "drop", "completion", "help"}

// matchCommand finds a command by prefix. Returns the command name or empty string if ambiguous/not found.
func matchCommand(input string) (string, []string) {
	// Check for exact match first
	for _, cmd := range commands {
		if cmd == input {
			return cmd, nil
		}
	}

	// Check for prefix match
	var matches []string
	for _, cmd := range commands {
		if strings.HasPrefix(cmd, input) {
			matches = append(matches, cmd)
		}
	}

	if len(matches) == 1 {
		return matches[0], nil
	}
	return "", matches
}

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	input := os.Args[1]

	// Handle help flags directly
	if input == "--help" || input == "-h" {
		printUsage()
		return
	}

	cmd, matches := matchCommand(input)
	if cmd == "" {
		if len(matches) > 0 {
			fmt.Fprintf(os.Stderr, "Ambiguous command '%s': %s\n", input, strings.Join(matches, ", "))
		} else {
			fmt.Fprintf(os.Stderr, "Unknown command: %s\n", input)
		}
		printUsage()
		os.Exit(1)
	}

	switch cmd {
	case "find":
		if len(os.Args) < 3 {
			fmt.Fprintln(os.Stderr, "Usage: giddy find <query> [--limit N] [--json] [--files-only]")
			os.Exit(1)
		}
		runSearch(os.Args[2:])
	case "up":
		runStart(os.Args[2:])
	case "down":
		runStop()
	case "bounce":
		runBounce(os.Args[2:])
	case "status":
		runStatus()
	case "health":
		runStats(os.Args[2:])
	case "log":
		runMonitor()
	case "init":
		runInit()
	case "mcp":
		runMcp()
	case "clean":
		runClean(os.Args[2:])
	case "drop":
		runDrop()
	case "completion":
		runCompletion(os.Args[2:])
	case "help":
		printUsage()
	}
}

func printUsage() {
	fmt.Println(`giddy - Semantic code search CLI

Usage:
  giddy find <query> [options]    Run semantic search
  giddy up [options]              Start the server
  giddy down                      Stop the server
  giddy bounce [options]          Restart the server
  giddy status                    Server status
  giddy health [options]          Diagnostic information
  giddy log                       Stream server logs
  giddy init                      Print setup prompt for new projects
  giddy mcp                       Run MCP server (for Claude Code)
  giddy drop                      Remove search index (keeps logs)
  giddy clean [options]           Remove all .giddyanne data
  giddy completion <shell>        Generate shell completions (bash, zsh, fish)

Commands can be abbreviated (e.g., 'giddy f' for 'find', 'giddy st' for 'status').

Find options:
  --limit N      Maximum results (default: 10)
  --json         Output as JSON
  --files-only   Only show file paths
  --verbose      Show full content in results
  --semantic     Vector search only (embeddings)
  --full-text    Keyword search only (BM25)
  --hybrid       Both combined (default)

Up/Bounce options:
  --port N       Preferred port (will find available if in use)
  --host ADDR    Host to bind to (default: from config)
  --verbose      Enable debug logging

Health options:
  --verbose      List all indexed files

Clean options:
  --force        Skip confirmation prompt`)
}

var errNoConfig = errors.New("no config found")

func printNoConfigHelp() {
	fmt.Fprintln(os.Stderr, `No .giddyanne.yaml found in current directory or parents.

To create one:

  giddy init    # Generates an LLM prompt to create config

Or manually create .giddyanne.yaml:

  paths:
    - path: src/
      description: Source code`)
}

func findProjectRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}

	for {
		configPath := filepath.Join(dir, ".giddyanne.yaml")
		if _, err := os.Stat(configPath); err == nil {
			return dir, nil
		}

		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}

	return "", errNoConfig
}

func getPIDFilePath(root, host string, port int) string {
	// Use same naming convention as Python (omit defaults)
	hostPart := host
	if host == defaultHost {
		hostPart = ""
	}
	portPart := ""
	if port != defaultPort {
		portPart = strconv.Itoa(port)
	}
	return filepath.Join(root, pidDir, fmt.Sprintf("%s-%s.pid", hostPart, portPart))
}

// findRunningServer scans pid files and returns (host, port, pid) or ("", 0, 0) if not running.
func findRunningServer(root string) (string, int, int) {
	dir := filepath.Join(root, pidDir)
	entries, err := os.ReadDir(dir)
	if err != nil {
		return "", 0, 0
	}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}

		name := entry.Name()
		if !strings.HasSuffix(name, ".pid") {
			continue
		}

		// Parse host-port from filename
		// Python omits defaults, so we may see: "-.pid", "-8080.pid", "localhost-.pid"
		stem := strings.TrimSuffix(name, ".pid")
		lastDash := strings.LastIndex(stem, "-")
		if lastDash == -1 {
			continue
		}

		hostPart := stem[:lastDash]
		portPart := stem[lastDash+1:]

		// Empty parts mean defaults were used
		host := hostPart
		if host == "" {
			host = defaultHost
		}
		port := defaultPort
		if portPart != "" {
			var err error
			port, err = strconv.Atoi(portPart)
			if err != nil {
				continue
			}
		}

		// Read PID from file
		pidPath := filepath.Join(dir, name)
		data, err := os.ReadFile(pidPath)
		if err != nil {
			continue
		}

		pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
		if err != nil {
			continue
		}

		// Check if process is running
		process, err := os.FindProcess(pid)
		if err != nil {
			continue
		}

		err = process.Signal(syscall.Signal(0))
		if err != nil {
			// Process not running, clean up stale PID file
			os.Remove(pidPath)
			continue
		}

		return host, port, pid
	}

	return "", 0, 0
}

func isServerHealthy(port int) bool {
	resp, err := http.Get(fmt.Sprintf("http://127.0.0.1:%d/health", port))
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}

func getGiddyDir() (string, error) {
	// Find where giddy binary is installed
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	exe, err = filepath.EvalSymlinks(exe)
	if err != nil {
		return "", err
	}
	// Binary is in the giddyanne repo root (or cmd/giddy during dev)
	dir := filepath.Dir(exe)
	// Check if we're in cmd/giddy (development)
	if filepath.Base(dir) == "giddy" && filepath.Base(filepath.Dir(dir)) == "cmd" {
		dir = filepath.Dir(filepath.Dir(dir))
	}
	return dir, nil
}

func startServer(root string, verbose bool, portOverride int, hostOverride string) (int, error) {
	// Find giddy installation directory
	giddyDir, err := getGiddyDir()
	if err != nil {
		return 0, fmt.Errorf("failed to find giddy installation: %w", err)
	}

	// Find python executable in giddy's venv
	pythonPath := filepath.Join(giddyDir, ".venv/bin/python")
	if _, err := os.Stat(pythonPath); err != nil {
		return 0, fmt.Errorf("python venv not found at %s (run install first)", pythonPath)
	}

	mainPath := filepath.Join(giddyDir, "main.py")
	if _, err := os.Stat(mainPath); err != nil {
		return 0, fmt.Errorf("main.py not found at %s", mainPath)
	}

	// Build command arguments
	args := []string{mainPath, "--path", root}
	if portOverride > 0 {
		args = append(args, "--port", strconv.Itoa(portOverride))
	}
	if hostOverride != "" {
		args = append(args, "--host", hostOverride)
	}
	if verbose {
		args = append(args, "--verbose")
	}

	// Always run as daemon
	args = append(args, "--daemon")
	cmd := exec.Command(pythonPath, args...)
	cmd.Dir = root

	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("failed to start server: %w", err)
	}

	// Wait for spawner to exit (it spawns the daemon then exits 0)
	if err := cmd.Wait(); err != nil {
		return 0, fmt.Errorf("server failed to start: %w", err)
	}

	// Poll for daemon health
	for {
		time.Sleep(pollInterval)
		if _, port, pid := findRunningServer(root); pid != 0 {
			if isServerHealthy(port) {
				return port, nil
			}
		}
	}
}

func ensureServer(root string) (int, error) {
	if _, port, pid := findRunningServer(root); pid != 0 {
		if isServerHealthy(port) {
			return port, nil
		}
	}
	return startServer(root, false, 0, "")
}

func runSearch(args []string) {
	var query string
	limit := 10
	jsonOutput := false
	filesOnly := false
	verbose := false
	searchType := "hybrid"

	// Parse arguments
	i := 0
	for i < len(args) {
		switch args[i] {
		case "--limit":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "--limit requires a value")
				os.Exit(1)
			}
			var err error
			limit, err = strconv.Atoi(args[i+1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "Invalid limit: %s\n", args[i+1])
				os.Exit(1)
			}
			i += 2
		case "--json":
			jsonOutput = true
			i++
		case "--files-only":
			filesOnly = true
			i++
		case "--verbose", "-v":
			verbose = true
			i++
		case "--semantic":
			searchType = "semantic"
			i++
		case "--full-text":
			searchType = "full-text"
			i++
		case "--hybrid":
			searchType = "hybrid"
			i++
		default:
			if query == "" {
				query = args[i]
			} else {
				query += " " + args[i]
			}
			i++
		}
	}

	if query == "" {
		fmt.Fprintln(os.Stderr, "No query provided")
		os.Exit(1)
	}

	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	port, err := ensureServer(root)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error starting server: %v\n", err)
		os.Exit(1)
	}

	// Make search request
	searchURL := fmt.Sprintf("http://127.0.0.1:%d/search", port)

	reqBody := SearchRequest{Query: query, Limit: limit, SearchType: searchType}
	reqBytes, err := json.Marshal(reqBody)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to encode request: %v\n", err)
		os.Exit(1)
	}

	resp, err := http.Post(searchURL, "application/json", strings.NewReader(string(reqBytes)))
	if err != nil {
		fmt.Fprintf(os.Stderr, "Search request failed: %v\n", err)
		os.Exit(1)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to read response: %v\n", err)
		os.Exit(1)
	}

	if resp.StatusCode != 200 {
		fmt.Fprintf(os.Stderr, "Search failed: %s\n", body)
		os.Exit(1)
	}

	var searchResp SearchResponse
	if err := json.Unmarshal(body, &searchResp); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse response: %v\n", err)
		os.Exit(1)
	}

	// Output results
	if jsonOutput {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		if filesOnly {
			// Strip content for --files-only --json
			type FileResult struct {
				Path      string `json:"path"`
				StartLine int    `json:"start_line"`
				EndLine   int    `json:"end_line"`
			}
			stripped := make([]FileResult, len(searchResp.Results))
			for i, r := range searchResp.Results {
				relPath := r.Path
				if strings.HasPrefix(r.Path, root) {
					relPath = strings.TrimPrefix(r.Path, root+"/")
				}
				stripped[i] = FileResult{Path: relPath, StartLine: r.StartLine, EndLine: r.EndLine}
			}
			enc.Encode(stripped)
		} else {
			enc.Encode(searchResp.Results)
		}
		return
	}

	if len(searchResp.Results) == 0 {
		fmt.Println("No results found.")
		return
	}

	for _, r := range searchResp.Results {
		// Make path relative to root
		relPath := r.Path
		if strings.HasPrefix(r.Path, root) {
			relPath = strings.TrimPrefix(r.Path, root+"/")
		}

		if filesOnly {
			fmt.Printf("%s:%d-%d\n", relPath, r.StartLine, r.EndLine)
		} else {
			fmt.Printf("\n%s:%d-%d (%.2f)\n", relPath, r.StartLine, r.EndLine, r.Score)
			lines := strings.Split(r.Content, "\n")
			if verbose {
				// Show all lines, no truncation
				for _, line := range lines {
					fmt.Printf("  %s\n", line)
				}
			} else {
				// Show first few lines of content
				maxLines := 3
				if len(lines) < maxLines {
					maxLines = len(lines)
				}
				for _, line := range lines[:maxLines] {
					if len(line) > 80 {
						line = line[:77] + "..."
					}
					fmt.Printf("  %s\n", line)
				}
				if len(lines) > 3 {
					fmt.Printf("  ... (%d more lines)\n", len(lines)-3)
				}
			}
		}
	}
}

func runStart(args []string) {
	verbose := false
	portOverride := 0
	hostOverride := ""

	// Parse arguments
	i := 0
	for i < len(args) {
		switch args[i] {
		case "--verbose", "-v":
			verbose = true
			i++
		case "--port":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "--port requires a value")
				os.Exit(1)
			}
			var err error
			portOverride, err = strconv.Atoi(args[i+1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "Invalid port: %s\n", args[i+1])
				os.Exit(1)
			}
			i += 2
		case "--host":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "--host requires a value")
				os.Exit(1)
			}
			hostOverride = args[i+1]
			i += 2
		default:
			fmt.Fprintf(os.Stderr, "Unknown option: %s\n", args[i])
			os.Exit(1)
		}
	}

	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	if host, port, pid := findRunningServer(root); pid != 0 {
		fmt.Printf("Server already running (PID %d, %s:%d)\n", pid, host, port)
		return
	}

	fmt.Println("Starting server...")
	port, err := startServer(root, verbose, portOverride, hostOverride)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Server started on port %d\n", port)
}

func runStop() {
	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	host, port, pid := findRunningServer(root)
	if pid == 0 {
		fmt.Println("Server not running")
		return
	}

	process, err := os.FindProcess(pid)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error finding process: %v\n", err)
		os.Exit(1)
	}

	if err := process.Signal(syscall.SIGTERM); err != nil {
		fmt.Fprintf(os.Stderr, "Error stopping server: %v\n", err)
		os.Exit(1)
	}

	// Wait for process to exit
	for i := 0; i < 30; i++ {
		time.Sleep(100 * time.Millisecond)
		if err := process.Signal(syscall.Signal(0)); err != nil {
			break
		}
	}

	os.Remove(getPIDFilePath(root, host, port))
	fmt.Println("Server stopped")
}

func runBounce(args []string) {
	verbose := false
	portOverride := 0
	hostOverride := ""

	// Parse arguments (same as runStart)
	i := 0
	for i < len(args) {
		switch args[i] {
		case "--verbose", "-v":
			verbose = true
			i++
		case "--port":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "--port requires a value")
				os.Exit(1)
			}
			var err error
			portOverride, err = strconv.Atoi(args[i+1])
			if err != nil {
				fmt.Fprintf(os.Stderr, "Invalid port: %s\n", args[i+1])
				os.Exit(1)
			}
			i += 2
		case "--host":
			if i+1 >= len(args) {
				fmt.Fprintln(os.Stderr, "--host requires a value")
				os.Exit(1)
			}
			hostOverride = args[i+1]
			i += 2
		default:
			fmt.Fprintf(os.Stderr, "Unknown option: %s\n", args[i])
			os.Exit(1)
		}
	}

	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	// Stop server if running
	if host, port, pid := findRunningServer(root); pid != 0 {
		fmt.Println("Stopping server...")
		process, err := os.FindProcess(pid)
		if err == nil {
			process.Signal(syscall.SIGTERM)
			// Wait for process to exit
			for i := 0; i < 30; i++ {
				time.Sleep(100 * time.Millisecond)
				if err := process.Signal(syscall.Signal(0)); err != nil {
					break
				}
			}
			os.Remove(getPIDFilePath(root, host, port))
		}
	}

	// Start server
	fmt.Println("Starting server...")
	port, err := startServer(root, verbose, portOverride, hostOverride)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Server started on port %d\n", port)
}

func runStatus() {
	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	host, port, pid := findRunningServer(root)
	if pid == 0 {
		fmt.Println("Not running")
		os.Exit(1)
	}

	// Query /status endpoint for detailed info
	resp, err := http.Get(fmt.Sprintf("http://%s:%d/status", host, port))
	if err != nil {
		// Can't connect - server is probably still starting up
		// Process is verified running by findRunningServer, so show "Starting..."
		fmt.Printf("Starting... (PID %d, %s:%d)\n", pid, host, port)
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		fmt.Printf("Starting... (PID %d, %s:%d)\n", pid, host, port)
		return
	}

	var status StatusResponse
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		fmt.Printf("Running (PID %d, %s:%d)\n", pid, host, port)
		return
	}

	switch status.State {
	case "indexing":
		fmt.Printf("Indexing: %d/%d files (%d%%)\n", status.Indexed, status.Total, status.Percent)
	case "ready":
		fmt.Printf("Running (PID %d, %s:%d)\n", pid, host, port)
	case "error":
		fmt.Printf("Error: %s\n", status.Error)
		os.Exit(1)
	default:
		fmt.Printf("Starting... (PID %d, %s:%d)\n", pid, host, port)
	}
}

func runStats(args []string) {
	verbose := false

	// Parse arguments
	for _, arg := range args {
		switch arg {
		case "--verbose", "-v":
			verbose = true
		default:
			fmt.Fprintf(os.Stderr, "Unknown option: %s\n", arg)
			os.Exit(1)
		}
	}

	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	host, port, pid := findRunningServer(root)
	if pid == 0 {
		fmt.Fprintln(os.Stderr, "Server not running (use 'giddy up' first)")
		os.Exit(1)
	}

	resp, err := http.Get(fmt.Sprintf("http://%s:%d/stats", host, port))
	if err != nil {
		// Server process is running but not responding - probably still starting
		fmt.Printf("Server starting... (PID %d, %s:%d)\n", pid, host, port)
		fmt.Println("Run 'giddy log' to watch startup progress")
		return
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		fmt.Fprintf(os.Stderr, "Failed to get stats: %s\n", body)
		os.Exit(1)
	}

	var stats StatsResponse
	if err := json.NewDecoder(resp.Body).Decode(&stats); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to parse stats: %v\n", err)
		os.Exit(1)
	}

	// Format index size
	sizeStr := formatBytes(stats.IndexSizeBytes)

	fmt.Printf("Indexed files: %d\n", stats.IndexedFiles)
	fmt.Printf("Total chunks:  %d\n", stats.TotalChunks)
	fmt.Printf("Index size:    %s\n", sizeStr)
	if stats.StartupDurationMs != nil {
		fmt.Printf("Startup time:  %.0f ms\n", *stats.StartupDurationMs)
	}
	if stats.AvgQueryLatencyMs != nil {
		fmt.Printf("Avg latency:   %.2f ms\n", *stats.AvgQueryLatencyMs)
	} else {
		fmt.Printf("Avg latency:   (no queries yet)\n")
	}

	if verbose && len(stats.Files) > 0 {
		fmt.Printf("\nFiles [chunks]:\n")
		for path, chunks := range stats.Files {
			// Make path relative to root if possible
			relPath := path
			if strings.HasPrefix(path, root) {
				relPath = strings.TrimPrefix(path, root+"/")
			}
			fmt.Printf("  %s [%d]\n", relPath, chunks)
		}
	}
}

func formatBytes(bytes int64) string {
	const (
		KB = 1024
		MB = KB * 1024
		GB = MB * 1024
	)
	switch {
	case bytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/GB)
	case bytes >= MB:
		return fmt.Sprintf("%.2f MB", float64(bytes)/MB)
	case bytes >= KB:
		return fmt.Sprintf("%.2f KB", float64(bytes)/KB)
	default:
		return fmt.Sprintf("%d bytes", bytes)
	}
}

func runMonitor() {
	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	host, port, pid := findRunningServer(root)
	if pid == 0 {
		fmt.Fprintln(os.Stderr, "Server not running (use 'giddy up' first)")
		os.Exit(1)
	}

	// Find the log file for this server (use same naming convention as Python)
	hostPart := host
	if host == defaultHost {
		hostPart = ""
	}
	portPart := ""
	if port != defaultPort {
		portPart = strconv.Itoa(port)
	}
	logPath := filepath.Join(root, ".giddyanne", fmt.Sprintf("%s-%s.log", hostPart, portPart))
	if _, err := os.Stat(logPath); os.IsNotExist(err) {
		fmt.Fprintf(os.Stderr, "Log file not found: %s\n", logPath)
		os.Exit(1)
	}

	fmt.Printf("Monitoring server (PID %d, %s:%d) - Ctrl+C to stop\n\n", pid, host, port)

	// Open and tail the log file
	file, err := os.Open(logPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to open log file: %v\n", err)
		os.Exit(1)
	}
	defer file.Close()

	// Print existing content
	buf := make([]byte, 4096)
	for {
		n, err := file.Read(buf)
		if n > 0 {
			os.Stdout.Write(buf[:n])
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading log: %v\n", err)
			os.Exit(1)
		}
	}

	// Tail new content
	for {
		n, err := file.Read(buf)
		if n > 0 {
			os.Stdout.Write(buf[:n])
		}
		if err != nil && err != io.EOF {
			fmt.Fprintf(os.Stderr, "Error reading log: %v\n", err)
			os.Exit(1)
		}

		// Check if server is still running
		if _, _, p := findRunningServer(root); p == 0 {
			fmt.Println("\nServer stopped")
			return
		}

		time.Sleep(100 * time.Millisecond)
	}
}

func runDrop() {
	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	// Warn if server is running (don't stop it)
	_, _, pid := findRunningServer(root)
	if pid != 0 {
		fmt.Println("Note: Server is running. Index will rebuild on restart.")
	}

	giddyanneDir := filepath.Join(root, ".giddyanne")
	removed := 0

	// Remove legacy vectors.lance (old location)
	legacyPath := filepath.Join(giddyanneDir, "vectors.lance")
	if _, err := os.Stat(legacyPath); err == nil {
		if err := os.RemoveAll(legacyPath); err != nil {
			fmt.Fprintf(os.Stderr, "Error removing legacy index: %v\n", err)
		} else {
			fmt.Println("Removed: vectors.lance (legacy)")
			removed++
		}
	}

	// Find and remove model-specific index directories
	entries, err := os.ReadDir(giddyanneDir)
	if err != nil {
		if !os.IsNotExist(err) {
			fmt.Fprintf(os.Stderr, "Error reading .giddyanne directory: %v\n", err)
		}
		if removed == 0 {
			fmt.Println("No index found.")
		}
		return
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		// Check if this directory contains vectors.lance
		vectorsPath := filepath.Join(giddyanneDir, entry.Name(), "vectors.lance")
		if _, err := os.Stat(vectorsPath); err == nil {
			if err := os.RemoveAll(vectorsPath); err != nil {
				fmt.Fprintf(os.Stderr, "Error removing %s: %v\n", entry.Name(), err)
			} else {
				fmt.Printf("Removed: %s/vectors.lance\n", entry.Name())
				removed++
			}
		}
	}

	if removed == 0 {
		fmt.Println("No index found.")
	} else {
		fmt.Println("Index dropped. Run 'giddy up' to rebuild.")
	}
}

func runClean(args []string) {
	force := false
	for _, arg := range args {
		switch arg {
		case "--force", "-f":
			force = true
		default:
			fmt.Fprintf(os.Stderr, "Unknown option: %s\n", arg)
			os.Exit(1)
		}
	}

	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	giddyanneDir := filepath.Join(root, ".giddyanne")

	// Check if directory exists
	if _, err := os.Stat(giddyanneDir); os.IsNotExist(err) {
		fmt.Println("Nothing to clean (.giddyanne not found).")
		return
	}

	// Require confirmation unless --force
	if !force {
		fmt.Print("Remove all giddyanne data (index, logs, everything)? [y/N] ")
		var response string
		fmt.Scanln(&response)
		if response != "y" && response != "Y" {
			fmt.Println("Cancelled.")
			return
		}
	}

	// Stop server if running
	_, _, pid := findRunningServer(root)
	if pid != 0 {
		fmt.Println("Stopping server...")
		runStop()
	}

	// Remove entire .giddyanne directory
	if err := os.RemoveAll(giddyanneDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error removing .giddyanne: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("Removed .giddyanne directory.")
}

func runInit() {
	// Check if config already exists
	cwd, err := os.Getwd()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error getting current directory: %v\n", err)
		os.Exit(1)
	}

	configPath := filepath.Join(cwd, ".giddyanne.yaml")
	if _, err := os.Stat(configPath); err == nil {
		fmt.Fprintln(os.Stderr, ".giddyanne.yaml already exists in this directory")
		os.Exit(1)
	}

	// Gather file listing
	files := gatherSourceFiles(cwd)

	// Print the prompt
	fmt.Println(`Copy and paste this prompt to an LLM to generate a config file:

---

Create a .giddyanne.yaml config file for my project. Here's the structure of my codebase:
`)
	fmt.Println(files)
	fmt.Println(`
The config format is:

` + "```yaml" + `
paths:
  - path: src/
    description: Brief description of what this directory contains
  - path: lib/
    description: Another description
` + "```" + `

Only files with supported extensions are indexed: .py, .go, .js, .jsx, .mjs, .ts, .tsx, .rs, .sql
Files matching .gitignore patterns are automatically excluded.

Group related code into paths with meaningful descriptions that explain the purpose of each area. These descriptions help semantic search understand context.

---`)
}

func runMcp() {
	// Find giddy installation directory
	giddyDir, err := getGiddyDir()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to find giddy installation: %v\n", err)
		os.Exit(1)
	}

	// Find python executable in giddy's venv
	pythonPath := filepath.Join(giddyDir, ".venv/bin/python")
	if _, err := os.Stat(pythonPath); err != nil {
		fmt.Fprintf(os.Stderr, "Python venv not found at %s (run install first)\n", pythonPath)
		os.Exit(1)
	}

	mcpMainPath := filepath.Join(giddyDir, "mcp_main.py")
	if _, err := os.Stat(mcpMainPath); err != nil {
		fmt.Fprintf(os.Stderr, "mcp_main.py not found at %s\n", mcpMainPath)
		os.Exit(1)
	}

	// Find project root for working directory
	root, err := findProjectRoot()
	if err != nil {
		if errors.Is(err, errNoConfig) {
			printNoConfigHelp()
		} else {
			fmt.Fprintf(os.Stderr, "Error finding project root: %v\n", err)
		}
		os.Exit(1)
	}

	// Run MCP server with stdio connected
	cmd := exec.Command(pythonPath, mcpMainPath)
	cmd.Dir = root
	cmd.Stdin = os.Stdin
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = append(os.Environ(), fmt.Sprintf("GIDDY_WATCH_PATH=%s", root))

	if err := cmd.Run(); err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			os.Exit(exitErr.ExitCode())
		}
		fmt.Fprintf(os.Stderr, "MCP server error: %v\n", err)
		os.Exit(1)
	}
}

func runCompletion(args []string) {
	if len(args) < 1 {
		fmt.Fprintln(os.Stderr, "Usage: giddy completion <bash|zsh|fish>")
		os.Exit(1)
	}

	shell := args[0]
	switch shell {
	case "bash":
		fmt.Print(bashCompletion)
	case "zsh":
		fmt.Print(zshCompletion)
	case "fish":
		fmt.Print(fishCompletion)
	default:
		fmt.Fprintf(os.Stderr, "Unknown shell: %s (supported: bash, zsh, fish)\n", shell)
		os.Exit(1)
	}
}

const bashCompletion = `# giddy bash completion
# Add to ~/.bashrc: eval "$(giddy completion bash)"

_giddy_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    local prev="${COMP_WORDS[COMP_CWORD-1]}"

    # Commands
    local commands="up down bounce find log status health init mcp clean drop completion help"

    # Complete command as first argument
    if [[ ${COMP_CWORD} -eq 1 ]]; then
        COMPREPLY=($(compgen -W "${commands}" -- "${cur}"))
        return
    fi

    # Command-specific completions
    case "${COMP_WORDS[1]}" in
        find|f)
            case "${prev}" in
                --limit)
                    COMPREPLY=()
                    return
                    ;;
            esac
            COMPREPLY=($(compgen -W "--limit --json --files-only --verbose --semantic --full-text --hybrid" -- "${cur}"))
            ;;
        up|u|bounce|bo)
            case "${prev}" in
                --port|--host)
                    COMPREPLY=()
                    return
                    ;;
            esac
            COMPREPLY=($(compgen -W "--port --host --verbose" -- "${cur}"))
            ;;
        health|he)
            COMPREPLY=($(compgen -W "--verbose" -- "${cur}"))
            ;;
        clean|cl)
            COMPREPLY=($(compgen -W "--force" -- "${cur}"))
            ;;
        completion|com)
            COMPREPLY=($(compgen -W "bash zsh fish" -- "${cur}"))
            ;;
    esac
}

complete -F _giddy_completions giddy
`

const zshCompletion = `#compdef giddy
# giddy zsh completion
# Add to ~/.zshrc: eval "$(giddy completion zsh)"

_giddy() {
    local -a commands
    commands=(
        'up:Start the server'
        'down:Stop the server'
        'bounce:Restart the server'
        'find:Run semantic search'
        'log:Stream server logs'
        'status:Server status'
        'health:Diagnostic information'
        'init:Print setup prompt for new projects'
        'mcp:Run MCP server (for Claude Code)'
        'drop:Remove search index (keeps logs)'
        'clean:Remove all .giddyanne data'
        'completion:Generate shell completions'
        'help:Show help'
    )

    local -a find_opts up_opts health_opts clean_opts completion_opts
    find_opts=(
        '--limit[Maximum results]:limit:'
        '--json[Output as JSON]'
        '--files-only[Only show file paths]'
        '--verbose[Show full content]'
        '--semantic[Vector search only]'
        '--full-text[Keyword search only]'
        '--hybrid[Both combined (default)]'
    )
    up_opts=(
        '--port[Server port]:port:'
        '--host[Host to bind]:host:'
        '--verbose[Enable debug logging]'
    )
    health_opts=(
        '--verbose[List all indexed files]'
    )
    clean_opts=(
        '--force[Skip confirmation prompt]'
    )
    completion_opts=(
        'bash:Generate bash completions'
        'zsh:Generate zsh completions'
        'fish:Generate fish completions'
    )

    _arguments -C \
        '1: :->command' \
        '*:: :->args'

    case $state in
        command)
            _describe -t commands 'giddy commands' commands
            ;;
        args)
            case ${words[1]} in
                find|f)
                    _arguments $find_opts
                    ;;
                up|u|bounce|bo)
                    _arguments $up_opts
                    ;;
                health|he)
                    _arguments $health_opts
                    ;;
                clean|cl)
                    _arguments $clean_opts
                    ;;
                completion|com)
                    _describe -t shells 'shells' completion_opts
                    ;;
            esac
            ;;
    esac
}

_giddy
`

const fishCompletion = `# giddy fish completion
# Add to ~/.config/fish/config.fish: giddy completion fish | source

# Disable file completion by default
complete -c giddy -f

# Commands
complete -c giddy -n '__fish_use_subcommand' -a up -d 'Start the server'
complete -c giddy -n '__fish_use_subcommand' -a down -d 'Stop the server'
complete -c giddy -n '__fish_use_subcommand' -a bounce -d 'Restart the server'
complete -c giddy -n '__fish_use_subcommand' -a find -d 'Run semantic search'
complete -c giddy -n '__fish_use_subcommand' -a log -d 'Stream server logs'
complete -c giddy -n '__fish_use_subcommand' -a status -d 'Server status'
complete -c giddy -n '__fish_use_subcommand' -a health -d 'Diagnostic information'
complete -c giddy -n '__fish_use_subcommand' -a init -d 'Print setup prompt for new projects'
complete -c giddy -n '__fish_use_subcommand' -a mcp -d 'Run MCP server (for Claude Code)'
complete -c giddy -n '__fish_use_subcommand' -a drop -d 'Remove search index (keeps logs)'
complete -c giddy -n '__fish_use_subcommand' -a clean -d 'Remove all .giddyanne data'
complete -c giddy -n '__fish_use_subcommand' -a completion -d 'Generate shell completions'
complete -c giddy -n '__fish_use_subcommand' -a help -d 'Show help'

# find options
complete -c giddy -n '__fish_seen_subcommand_from find f' -l limit -d 'Maximum results' -r
complete -c giddy -n '__fish_seen_subcommand_from find f' -l json -d 'Output as JSON'
complete -c giddy -n '__fish_seen_subcommand_from find f' -l files-only -d 'Only show file paths'
complete -c giddy -n '__fish_seen_subcommand_from find f' -l verbose -d 'Show full content'
complete -c giddy -n '__fish_seen_subcommand_from find f' -l semantic -d 'Vector search only'
complete -c giddy -n '__fish_seen_subcommand_from find f' -l full-text -d 'Keyword search only'
complete -c giddy -n '__fish_seen_subcommand_from find f' -l hybrid -d 'Both combined (default)'

# up/bounce options
complete -c giddy -n '__fish_seen_subcommand_from up u bounce bo' -l port -d 'Server port' -r
complete -c giddy -n '__fish_seen_subcommand_from up u bounce bo' -l host -d 'Host to bind' -r
complete -c giddy -n '__fish_seen_subcommand_from up u bounce bo' -l verbose -d 'Enable debug logging'

# health options
complete -c giddy -n '__fish_seen_subcommand_from health he' -l verbose -d 'List all indexed files'

# clean options
complete -c giddy -n '__fish_seen_subcommand_from clean cl' -l force -d 'Skip confirmation prompt'

# completion shells
complete -c giddy -n '__fish_seen_subcommand_from completion com' -a 'bash zsh fish' -d 'Shell type'
`

func gatherSourceFiles(root string) string {
	var files []string
	extensions := map[string]bool{
		".py": true, ".js": true, ".ts": true, ".tsx": true, ".jsx": true,
		".go": true, ".rs": true, ".java": true, ".rb": true, ".md": true,
		".c": true, ".cpp": true, ".h": true, ".hpp": true, ".cs": true,
		".swift": true, ".kt": true, ".scala": true, ".php": true,
	}
	ignoreDirs := map[string]bool{
		"node_modules": true, "__pycache__": true, ".git": true, ".venv": true,
		"venv": true, "vendor": true, "dist": true, "build": true, ".next": true,
		"target": true, ".idea": true, ".vscode": true,
	}

	filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}

		// Skip ignored directories
		if info.IsDir() {
			if ignoreDirs[info.Name()] {
				return filepath.SkipDir
			}
			return nil
		}

		// Check extension
		ext := filepath.Ext(path)
		if !extensions[ext] {
			return nil
		}

		// Get relative path
		relPath, err := filepath.Rel(root, path)
		if err != nil {
			return nil
		}

		files = append(files, relPath)

		// Limit to 50 files
		if len(files) >= 50 {
			return filepath.SkipAll
		}
		return nil
	})

	return strings.Join(files, "\n")
}
