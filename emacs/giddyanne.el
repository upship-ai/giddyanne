;;; giddyanne.el --- Semantic codebase search -*- lexical-binding: t; -*-

;; Copyright (C) 2025 Joey

;; Author: Joey
;; Version: 1.6.1
;; Package-Requires: ((emacs "28.1"))
;; Keywords: tools, search, semantic
;; URL: https://github.com/mazziv/giddyanne

;; This file is not part of GNU Emacs.

;;; Commentary:

;; Semantic codebase search powered by giddyanne.
;; Indexes files with embeddings, lets you find code by meaning.
;;
;; Commands:
;;   giddyanne-find    - Search with semantic matching
;;   giddyanne-up      - Start the server
;;   giddyanne-down    - Stop the server
;;   giddyanne-status  - Show server status (one-line)
;;   giddyanne-health  - Show index statistics (multi-line)
;;   giddyanne-sitemap - Show indexed files as tree
;;   giddyanne-log     - Toggle log buffer
;;
;; Setup for Doom Emacs:
;; (use-package giddyanne
;;   :load-path "~/projects/giddyanne/emacs"
;;   :commands (giddyanne-find giddyanne-up giddyanne-down giddyanne-status giddyanne-log giddyanne-health giddyanne-sitemap)
;;   :init
;;   (map! :leader
;;         (:prefix "s"
;;          :desc "giddyanne" "g" #'giddyanne-find)
;;         (:prefix "g"
;;                  (:prefix ("a" . "giddyanne")
;;                   :desc "up      Starts the server" "u" #'giddyanne-up
;;                   :desc "down    Stops the server" "d" #'giddyanne-down
;;                   :desc "find    Run semantic search" "f" #'giddyanne-find
;;                   :desc "log     It's log, log, log!" "l" #'giddyanne-log
;;                   :desc "status  Server status" "s" #'giddyanne-status
;;                   :desc "health  Diagnostic information" "h" #'giddyanne-health))))

;;; Code:

(require 'json)
(require 'cl-lib)
(require 'pulse)

(defgroup giddyanne nil
  "Semantic codebase search."
  :group 'tools
  :prefix "giddyanne-")

(defcustom giddyanne-executable "giddy"
  "Path to the giddy executable."
  :type 'string
  :group 'giddyanne)

(defcustom giddyanne-vertico-group-format
  (when (boundp 'vertico-group-format)
    #("  %s  " 0 6 (face nil)))
  "Format for Vertico group headers during giddyanne search.
Set to nil to use Vertico's default. The default removes dashes."
  :type '(choice string (const nil))
  :group 'giddyanne)

;;; Utilities

(defvar giddyanne--results nil
  "Alist mapping candidate strings to result plists.")

(defun giddyanne--executable ()
  "Return path to giddy executable, or signal error if not found."
  (or (executable-find giddyanne-executable)
      (user-error "giddy not found in PATH. Install from https://github.com/mazziv/giddyanne")))

(defun giddyanne--run (&rest args)
  "Run giddy with ARGS and return output as string."
  (let ((executable (giddyanne--executable)))
    (with-temp-buffer
      (let ((exit-code (apply #'call-process executable nil t nil args)))
        (if (zerop exit-code)
            (buffer-string)
          (user-error "giddyanne: %s" (string-trim (buffer-string))))))))

(defun giddyanne--run-async (args callback)
  "Run giddy with ARGS asynchronously, call CALLBACK with output."
  (let ((executable (giddyanne--executable)))
    (make-process
     :name "giddyanne"
     :buffer nil
     :command (cons executable args)
     :sentinel (lambda (proc event)
                 (when (string-match-p "finished" event)
                   (funcall callback))))))

;;; Find

(defun giddyanne--find (query)
  "Run semantic search for QUERY, return list of result plists."
  (let* ((output (giddyanne--run "find" "--json" query))
         (json-array-type 'list)
         (json-object-type 'plist)
         (json-key-type 'keyword))
    (condition-case nil
        (json-read-from-string output)
      (error nil))))

(defun giddyanne--format-candidate (result)
  "Format RESULT plist as a candidate string for display."
  (let* ((path (plist-get result :path))
         (start (plist-get result :start_line))
         (content (plist-get result :content))
         (preview (string-trim (car (split-string (or content "") "\n"))))
         (preview-truncated (if (> (length preview) 80)
                                (concat (substring preview 0 77) "...")
                              preview))
         ;; Make path relative to project root if possible
         (project-root (and (fboundp 'projectile-project-root)
                            (projectile-project-p)
                            (projectile-project-root)))
         (relative-path (if (and project-root
                                 (string-prefix-p project-root path))
                            (file-relative-name path project-root)
                          path))
         ;; Format: "LINE:CONTENT" with faded line number
         (line-prefix (propertize (format "%d: " start) 'face 'shadow))
         (candidate (concat line-prefix preview-truncated)))
    ;; Store the file path as a text property for grouping
    (add-text-properties 0 (length candidate)
                         (list 'giddyanne-file relative-path
                               'giddyanne-result result)
                         candidate)
    candidate))

(defun giddyanne--goto-result (candidate)
  "Jump to the file and line for CANDIDATE."
  ;; Get result from text property or fall back to lookup
  (let ((result (or (get-text-property 0 'giddyanne-result candidate)
                    (cdr (assoc candidate giddyanne--results))
                    (cdar giddyanne--results))))
    (when result
      (let ((path (plist-get result :path))
            (start-line (plist-get result :start_line))
            (end-line (plist-get result :end_line)))
        (find-file path)
        (goto-char (point-min))
        (forward-line (1- start-line))
        ;; Highlight the range
        (pulse-momentary-highlight-region
         (point)
         (save-excursion
           (forward-line (- end-line start-line))
           (end-of-line)
           (point)))))))

(defun giddyanne--file-icon (filename)
  "Return a file type icon for FILENAME, or empty string if unavailable."
  (if (fboundp 'nerd-icons-icon-for-file)
      (concat (nerd-icons-icon-for-file filename) (propertize " - " 'face 'shadow))
    ""))

(defun giddyanne--group-function (candidate transform)
  "Group CANDIDATE by file path. If TRANSFORM, return candidate unchanged."
  (if transform
      candidate
    ;; Return: [icon] bold-name ────────
    (let* ((file (get-text-property 0 'giddyanne-file candidate))
           (icon (giddyanne--file-icon file))
           (name (propertize file 'face 'highlight)))
      (concat icon name))))

;;; Commands

;;;###autoload
(defun giddyanne-find (query)
  "Search project with giddyanne semantic search for QUERY."
  (interactive "sGiddyanne find: ")
  (when (string-empty-p query)
    (user-error "Empty search query"))
  (let* ((results (giddyanne--find query))
         (sorted (sort results
                       (lambda (a b)
                         (> (plist-get a :score)
                            (plist-get b :score))))))
    (unless sorted
      (user-error "No results for: %s" query))
    (setq giddyanne--results
          (mapcar (lambda (r)
                    (cons (giddyanne--format-candidate r) r))
                  sorted))
    (let* ((candidates (mapcar #'car giddyanne--results))
           ;; Override Vertico group format to remove dashes
           ;; (vertico-group-format (or giddyanne-vertico-group-format
           ;;                           (and (boundp 'vertico-group-format)
           ;;                                vertico-group-format)))
           (choice (completing-read
                    (format "Results for \"%s\": " query)
                    (lambda (str pred action)
                      (pcase action
                        ('metadata
                         `(metadata
                           (display-sort-function . identity)
                           (cycle-sort-function . identity)
                           (category . giddyanne)
                           (group-function . giddyanne--group-function)))
                        ('t candidates)
                        (_ (complete-with-action action candidates str pred)))))))
      (when (and choice (not (string-empty-p choice)))
        (giddyanne--goto-result choice)))))

;;; Log

(defconst giddyanne-log-buffer-name "*Giddyanne Log*"
  "Name of the giddyanne log buffer.")

(defvar giddyanne--log-process nil
  "The log process.")

(defun giddyanne--log-running-p ()
  "Return t if log process is running."
  (and giddyanne--log-process
       (process-live-p giddyanne--log-process)))

(defun giddyanne--log-start ()
  "Start the log process."
  (let ((buf (get-buffer-create giddyanne-log-buffer-name))
        (executable (giddyanne--executable)))
    (with-current-buffer buf
      (let ((inhibit-read-only t))
        (erase-buffer))
      (special-mode)
      (when (bound-and-true-p evil-mode)
        (evil-local-set-key 'normal (kbd "<escape>") #'quit-window))
      (add-hook 'kill-buffer-hook #'giddyanne--log-stop nil t))
    (setq giddyanne--log-process
          (make-process
           :name "giddyanne-log"
           :buffer buf
           :command (list executable "log")
           :filter (lambda (proc output)
                     (when (buffer-live-p (process-buffer proc))
                       (with-current-buffer (process-buffer proc)
                         (let ((inhibit-read-only t))
                           (goto-char (point-max))
                           (insert output)))))
           :sentinel (lambda (proc event)
                       (when (buffer-live-p (process-buffer proc))
                         (with-current-buffer (process-buffer proc)
                           (let ((inhibit-read-only t))
                             (goto-char (point-max))
                             (insert (format "\n[Process %s]" (string-trim event)))))))))))

(defun giddyanne--log-stop ()
  "Stop the log process."
  (when (giddyanne--log-running-p)
    (kill-process giddyanne--log-process)
    (setq giddyanne--log-process nil)))

;;;###autoload
(defun giddyanne-log ()
  "Toggle the giddyanne log buffer."
  (interactive)
  (let ((buf (get-buffer giddyanne-log-buffer-name))
        (win (get-buffer-window giddyanne-log-buffer-name)))
    (cond
     ;; Buffer visible - hide it
     (win
      (quit-window nil win))
     ;; Buffer exists but not visible - show it
     (buf
      (pop-to-buffer buf))
     ;; No buffer - start log and show
     (t
      (giddyanne--log-start)
      (pop-to-buffer giddyanne-log-buffer-name)))))

;;;###autoload
(defun giddyanne-up ()
  "Start the giddyanne server."
  (interactive)
  (message "Starting giddyanne server...")
  (giddyanne--run-async
   '("up")
   (lambda ()
     (message "giddyanne server started"))))

;;;###autoload
(defun giddyanne-down ()
  "Stop the giddyanne server."
  (interactive)
  (let ((output (giddyanne--run "down")))
    (message "%s" (string-trim output))))

;;;###autoload
(defun giddyanne-status ()
  "Show giddyanne server status."
  (interactive)
  (let ((output (giddyanne--run "status")))
    (message "%s" (string-trim output))))

;;;###autoload
(defun giddyanne-health ()
  "Show giddyanne index statistics."
  (interactive)
  (let ((output (giddyanne--run "health")))
    (with-current-buffer (get-buffer-create "*Giddyanne Health*")
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (string-trim output)))
      (goto-char (point-min))
      (special-mode)
      (when (bound-and-true-p evil-mode)
        (evil-local-set-key 'normal (kbd "<escape>") #'quit-window))
      (pop-to-buffer (current-buffer)))))

;;;###autoload
(defun giddyanne-sitemap ()
  "Show giddyanne indexed files as a tree."
  (interactive)
  (let ((output (giddyanne--run "sitemap")))
    (with-current-buffer (get-buffer-create "*Giddyanne Sitemap*")
      (let ((inhibit-read-only t))
        (erase-buffer)
        (insert (string-trim output)))
      (goto-char (point-min))
      (special-mode)
      (when (bound-and-true-p evil-mode)
        (evil-local-set-key 'normal (kbd "<escape>") #'quit-window))
      (pop-to-buffer (current-buffer)))))

(provide 'giddyanne)
;;; giddyanne.el ends here
