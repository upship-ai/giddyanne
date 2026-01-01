import * as vscode from 'vscode';
import * as path from 'path';
import { GiddyanneClient, SearchResult } from './client';

const client = new GiddyanneClient();

// Decoration for highlighting search results (like Emacs pulse)
const resultHighlight = vscode.window.createTextEditorDecorationType({
    backgroundColor: new vscode.ThemeColor('editor.findMatchHighlightBackground'),
    isWholeLine: true
});

interface ResultQuickPickItem extends vscode.QuickPickItem {
    result: SearchResult;
}

function formatResult(result: SearchResult, workspaceRoot: string): ResultQuickPickItem {
    const relativePath = result.path.startsWith(workspaceRoot)
        ? path.relative(workspaceRoot, result.path)
        : result.path;

    const preview = result.content.split('\n')[0].trim();
    const truncated = preview.length > 80 ? preview.substring(0, 77) + '...' : preview;

    return {
        label: `$(file) ${relativePath}`,
        description: `L${result.start_line}`,
        detail: truncated,
        result
    };
}

function groupResultsByFile(items: ResultQuickPickItem[]): vscode.QuickPickItem[] {
    const grouped: vscode.QuickPickItem[] = [];
    let currentFile = '';

    for (const item of items) {
        const file = item.result.path;
        if (file !== currentFile) {
            // Add separator for new file group
            if (grouped.length > 0) {
                grouped.push({ label: '', kind: vscode.QuickPickItemKind.Separator });
            }
            currentFile = file;
        }
        grouped.push(item);
    }

    return grouped;
}

async function gotoResult(result: SearchResult): Promise<void> {
    const uri = vscode.Uri.file(result.path);
    const doc = await vscode.workspace.openTextDocument(uri);
    const editor = await vscode.window.showTextDocument(doc);

    // Go to start line
    const startLine = Math.max(0, result.start_line - 1);
    const endLine = Math.max(0, result.end_line - 1);

    const range = new vscode.Range(startLine, 0, endLine, 0);
    editor.revealRange(range, vscode.TextEditorRevealType.InCenter);

    // Set cursor
    editor.selection = new vscode.Selection(startLine, 0, startLine, 0);

    // Highlight the range briefly (like Emacs pulse)
    const highlightRange = new vscode.Range(startLine, 0, endLine + 1, 0);
    editor.setDecorations(resultHighlight, [highlightRange]);

    // Remove highlight after 1.5 seconds
    setTimeout(() => {
        editor.setDecorations(resultHighlight, []);
    }, 1500);
}

export async function find(): Promise<void> {
    const query = await vscode.window.showInputBox({
        prompt: 'Giddyanne find',
        placeHolder: 'Enter semantic search query...'
    });

    if (!query) {
        return;
    }

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    try {
        const results = await vscode.window.withProgress(
            {
                location: vscode.ProgressLocation.Notification,
                title: 'Searching...',
                cancellable: false
            },
            async () => client.find(query)
        );

        if (results.length === 0) {
            vscode.window.showInformationMessage(`No results for: ${query}`);
            return;
        }

        // Sort by score descending
        results.sort((a, b) => b.score - a.score);

        // Format as QuickPick items
        const items = results.map(r => formatResult(r, workspaceFolder.uri.fsPath));
        const grouped = groupResultsByFile(items);

        const selected = await vscode.window.showQuickPick(grouped, {
            title: `Results for "${query}"`,
            placeHolder: 'Select a result to open',
            matchOnDescription: true,
            matchOnDetail: true
        });

        if (selected && 'result' in selected) {
            await gotoResult((selected as ResultQuickPickItem).result);
        }
    } catch (error: any) {
        vscode.window.showErrorMessage(`Giddyanne: ${error.message}`);
    }
}

export async function up(): Promise<void> {
    try {
        vscode.window.showInformationMessage('Starting giddyanne server...');
        await client.up();
        vscode.window.showInformationMessage('Giddyanne server started');
    } catch (error: any) {
        vscode.window.showErrorMessage(`Giddyanne: ${error.message}`);
    }
}

export function down(): void {
    try {
        const output = client.down();
        vscode.window.showInformationMessage(output);
    } catch (error: any) {
        vscode.window.showErrorMessage(`Giddyanne: ${error.message}`);
    }
}

export function status(): void {
    try {
        const output = client.status();
        vscode.window.showInformationMessage(`Giddyanne: ${output}`);
    } catch (error: any) {
        vscode.window.showErrorMessage(`Giddyanne: ${error.message}`);
    }
}

export async function health(): Promise<void> {
    try {
        const output = client.health();

        // Show in output channel for multi-line content
        const channel = vscode.window.createOutputChannel('Giddyanne Health');
        channel.clear();
        channel.appendLine(output);
        channel.show();
    } catch (error: any) {
        vscode.window.showErrorMessage(`Giddyanne: ${error.message}`);
    }
}

let logTerminal: vscode.Terminal | undefined;

export function log(): void {
    // Reuse existing terminal if still open
    if (logTerminal && logTerminal.exitStatus === undefined) {
        logTerminal.show();
        return;
    }

    const executable = client.getExecutablePath();
    logTerminal = vscode.window.createTerminal({
        name: 'Giddyanne Log',
        shellPath: executable,
        shellArgs: ['log']
    });
    logTerminal.show();
}
