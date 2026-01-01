import * as vscode from 'vscode';
import * as commands from './commands';

export function activate(context: vscode.ExtensionContext): void {
    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('giddyanne.find', commands.find),
        vscode.commands.registerCommand('giddyanne.up', commands.up),
        vscode.commands.registerCommand('giddyanne.down', commands.down),
        vscode.commands.registerCommand('giddyanne.status', commands.status),
        vscode.commands.registerCommand('giddyanne.health', commands.health),
        vscode.commands.registerCommand('giddyanne.log', commands.log)
    );

    console.log('Giddyanne extension activated');
}

export function deactivate(): void {
    // Cleanup if needed
}
