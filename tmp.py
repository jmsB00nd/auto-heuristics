def iterative_heuristic_search(self, num_iterations, base_prompt):
        """
        Iteratively generates, evaluates, and refines a heuristic cost function.
        Appends the history of past attempts and their scores to the prompt 
        to ensure the LLM learns from previous mistakes and successes.
        """
        console.print(Rule("ITERATIVE HEURISTIC SEARCH", style="bold cyan"))
        console.print(_Panel(
            f"[bold cyan]Starting {num_iterations} iterations of iterative generation & evaluation...[/bold cyan]",
            border_style="cyan",
        ))

        # Ensure directories exist
        iterative_log_dir = os.path.join(LOG_DIR, "iterative_search")
        os.makedirs(iterative_log_dir, exist_ok=True)
        os.makedirs("heuristics", exist_ok=True)

        current_prompt = base_prompt
        all_results = []

        for i in range(1, num_iterations + 1):
            console.print(Rule(f"Iteration {i} / {num_iterations}", style="bold magenta"))

            response = self.query_llm(current_prompt, reset_conversation=True)
            save_log(iterative_log_dir, f"iteration_{i}_response.txt", response or "")

            if not response:
                console.print(f"[bold red]Failed to get a response from LLM on iteration {i}. Skipping.[/bold red]")
                continue

            strategy_match = re.search(r'STRATEGY:\s*(.*?)(?=\nINTUITION|\nCODE|\n|$)', response, re.IGNORECASE)
            intuition_match = re.search(r'INTUITION:\s*(.*?)(?=\nCODE|\nSTRATEGY|\n|$)', response, re.IGNORECASE | re.DOTALL)

            strategy = strategy_match.group(1).strip() if strategy_match else f"Generated_Idea_{i}"
            intuition = intuition_match.group(1).strip() if intuition_match else "No intuition parsed."

            console.print(f"[bold cyan]Generated Idea:[/bold cyan] {strategy}")
            console.print(f"[bold cyan]Intuition:[/bold cyan] {intuition[:200]}{'...' if len(intuition) > 200 else ''}")

            code = self.parse_response(response)

            if not code:
                console.print("[bold red]✘ Failed to extract Python code. Updating history and skipping to next iteration.[/bold red]")
                current_prompt += f"\n\n### History - Iteration {i}\nStatus: Failed to extract valid python code.\n"
                continue

            safe_strategy = re.sub(r'[^\w\-]', '_', strategy)
            file_name = f"heuristics/iter_search_{i}_{safe_strategy}.py"
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(f"# Strategy: {strategy}\n# Intuition: {intuition}\n\n{code}")
            
            console.print(f"[dim]Code extracted and saved to [bold]{file_name}[/bold][/dim]")

            console.print("[bold yellow]Evaluating heuristic...[/bold yellow]")
            stats = self.inject_and_run(code, timeout_seconds=self.timeout_seconds)

            succeeded = not stats.get('error')
            mean_swaps = stats.get('mean_swaps', float('inf'))
            mean_depth = stats.get('mean_depth', 0)

            if succeeded:
                console.print(
                    f"[bold green]✔ Iteration {i} SUCCESS[/bold green]\n"
                    f"  Avg Swaps: {mean_swaps:.2f} | Avg Depth: {mean_depth:.2f}"
                )
                status_text = f"SUCCESS - Swaps: {mean_swaps:.2f}, Depth: {mean_depth:.2f}"
            else:
                error_msg = stats.get('error', 'Unknown Error')
                console.print(f"[bold red]✘ Iteration {i} FAILED: {error_msg}[/bold red]")
                status_text = f"FAILED - Error: {error_msg}"

            current_prompt += (
                f"\n\n### History - Iteration {i}\n"
                f"STRATEGY: {strategy}\n"
                f"INTUITION: {intuition}\n"
                f"RESULT: {status_text}\n"
            )

            result_entry = {
                "iteration": i,
                "strategy": strategy,
                "intuition": intuition,
                "status": "SUCCESS" if succeeded else "FAILED",
                "mean_swaps": mean_swaps,
                "mean_depth": mean_depth,
                "error": stats.get('error'),
                "file_path": file_name
            }
            all_results.append(result_entry)
            save_json(iterative_log_dir, f"results_up_to_{i}.json", all_results)

        console.print(Rule("Iterative Search Complete", style="bold white"))
        successful_runs = [r for r in all_results if r["status"] == "SUCCESS"]
        
        if successful_runs:
            best_run = min(successful_runs, key=lambda x: x["mean_swaps"])
            console.print(_Panel(
                f"[bold green]Best Strategy:[/bold green] {best_run['strategy']} (Iteration {best_run['iteration']})\n"
                f"[bold green]Mean Swaps:[/bold green] {best_run['mean_swaps']:.2f}\n"
                f"[bold green]Mean Depth:[/bold green] {best_run['mean_depth']:.2f}\n"
                f"[bold green]Saved at:[/bold green] {best_run['file_path']}",
                title="[bold]Iterative Pipeline Winner[/bold]",
                border_style="green"
            ))
        else:
            console.print("[bold red]No successful heuristics were generated during this run.[/bold red]")

        return all_results
