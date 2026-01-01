import * as vscode from 'vscode';
import * as http from 'http';
import * as fs from 'fs';
import * as path from 'path';
import { execSync, spawn } from 'child_process';

export interface SearchResult {
    path: string;
    content: string;
    start_line: number;
    end_line: number;
    score: number;
    description?: string;
}

export interface HealthInfo {
    status: string;
    files: number;
    chunks: number;
    last_indexed?: string;
}

/**
 * Client for communicating with giddyanne server.
 * Prefers HTTP API, falls back to CLI.
 */
export class GiddyanneClient {
    getExecutablePath(): string {
        const config = vscode.workspace.getConfiguration('giddyanne');
        const configured = config.get<string>('executable', 'giddy');

        // If absolute path or explicitly configured, use as-is
        if (configured.startsWith('/') || configured !== 'giddy') {
            return configured;
        }

        // Search common locations (VSCode doesn't inherit shell PATH)
        const home = process.env.HOME || '';
        const candidates = [
            path.join(home, 'bin', 'giddy'),
            path.join(home, '.local', 'bin', 'giddy'),
            '/usr/local/bin/giddy',
            '/opt/homebrew/bin/giddy'
        ];

        for (const candidate of candidates) {
            if (fs.existsSync(candidate)) {
                return candidate;
            }
        }

        // Fall back to bare name (might work if in system PATH)
        return configured;
    }

    private getServerUrl(): string | undefined {
        const config = vscode.workspace.getConfiguration('giddyanne');
        const configured = config.get<string>('serverUrl');
        if (configured) {
            return configured;
        }

        // Try to read port from .giddyanne/port file
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        if (workspaceFolder) {
            const portFile = path.join(workspaceFolder.uri.fsPath, '.giddyanne', 'port');
            try {
                const port = fs.readFileSync(portFile, 'utf8').trim();
                return `http://127.0.0.1:${port}`;
            } catch {
                // Port file doesn't exist, server not running
            }
        }
        return undefined;
    }

    private async httpGet<T>(endpoint: string): Promise<T> {
        const baseUrl = this.getServerUrl();
        if (!baseUrl) {
            throw new Error('Server not running');
        }

        return new Promise((resolve, reject) => {
            const url = `${baseUrl}${endpoint}`;
            http.get(url, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    if (res.statusCode !== 200) {
                        reject(new Error(`HTTP ${res.statusCode}: ${data}`));
                        return;
                    }
                    try {
                        resolve(JSON.parse(data));
                    } catch {
                        reject(new Error(`Invalid JSON: ${data}`));
                    }
                });
            }).on('error', reject);
        });
    }

    private runCli(...args: string[]): string {
        const executable = this.getExecutablePath();
        const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
        const cwd = workspaceFolder?.uri.fsPath;

        try {
            return execSync(`${executable} ${args.join(' ')}`, {
                encoding: 'utf8',
                cwd,
                timeout: 30000
            }).trim();
        } catch (error: any) {
            throw new Error(error.stderr || error.message);
        }
    }

    private runCliAsync(...args: string[]): Promise<string> {
        return new Promise((resolve, reject) => {
            const executable = this.getExecutablePath();
            const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
            const cwd = workspaceFolder?.uri.fsPath;

            const proc = spawn(executable, args, { cwd });
            let stdout = '';
            let stderr = '';

            proc.stdout.on('data', data => stdout += data);
            proc.stderr.on('data', data => stderr += data);
            proc.on('close', code => {
                if (code === 0) {
                    resolve(stdout.trim());
                } else {
                    reject(new Error(stderr || `Exit code ${code}`));
                }
            });
            proc.on('error', reject);
        });
    }

    async find(query: string, limit: number = 10): Promise<SearchResult[]> {
        // Try HTTP first
        const serverUrl = this.getServerUrl();
        if (serverUrl) {
            try {
                const encoded = encodeURIComponent(query);
                return await this.httpGet<SearchResult[]>(
                    `/search?q=${encoded}&limit=${limit}`
                );
            } catch {
                // Fall through to CLI
            }
        }

        // Fall back to CLI
        const output = this.runCli('find', '--json', `"${query}"`, `--limit`, String(limit));
        return JSON.parse(output);
    }

    async up(): Promise<void> {
        await this.runCliAsync('up');
    }

    down(): string {
        return this.runCli('down');
    }

    status(): string {
        return this.runCli('status');
    }

    health(): string {
        return this.runCli('health');
    }

    isServerRunning(): boolean {
        return this.getServerUrl() !== undefined;
    }
}
