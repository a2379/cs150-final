<script lang="ts">
	import { onMount } from 'svelte';

	export let gridState: Array<Array<number>>;

	function cellSelected(gridIndex: number, i: number) {
		if (gridState[gridIndex][i] == 0.0) gridState[gridIndex][i] = Math.random();
		else gridState[gridIndex][i] = 0.0;
	}

	function randomizeGrid() {
		for (let i = 0; i < gridState.length; i++) {
			for (let j = 0; j < gridState[i].length; j++) {
				if (Math.random() < 0.5) gridState[i][j] = Math.random();
				else gridState[i][j] = 0.0;
			}
		}
	}

	function clearGrid() {
		gridState = [new Array(16).fill(0.0), new Array(16).fill(0.0)];
	}

	onMount(() => {
		randomizeGrid();
	});
</script>

<div class="flex h-full w-full flex-col items-center justify-center overflow-hidden">
	<div class="flex h-4/5 w-4/5 flex-col justify-center space-y-8">
		{#each gridState as grid, gridIndex}
			<div class="flex h-[4vh] w-full justify-evenly">
				{#each grid as _, i}
					<button
						aria-label={`${i}`}
						type="button"
						class={`btn ${gridState[gridIndex][i] == 0.0 ? 'btn-dash' : ''} btn-success h-full w-[4vh]`}
						onclick={() => cellSelected(gridIndex, i)}
						value={i}
					>
					</button>
				{/each}
			</div>
		{/each}
	</div>
	<input type="hidden" name="gridState" value={JSON.stringify(gridState)} />

	<div class="flex h-1/5 w-full items-center justify-center space-x-8">
		<button type="button" class="btn btn-soft btn-error" onclick={() => clearGrid()}
			>Clear Grid</button
		>
		<button type="button" class="btn btn-soft btn-success" onclick={() => randomizeGrid()}
			>Randomize Grid</button
		>
	</div>
</div>
