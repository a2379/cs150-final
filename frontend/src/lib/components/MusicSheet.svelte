<script lang="ts">
	import { onMount } from 'svelte';
	import TimeSignature from './TimeSignature.svelte';
	import TrebleClef from './TrebleClef.svelte';
	import Quarter from './Quarter.svelte';

	let selectedNotes: Set<string> = $state(new Set());

	let notes: Array<number> = new Array(16).fill(0);
	const columnWidth: number = 100 / 16;

	let hoverX: number = $state(-1);
	let hoverY: number = $state(-1);

	function mouseEnterSheet(i: number, j: number) {
		hoverX = i;
		hoverY = j;
	}

	function mouseLeaveSheet() {
		hoverX = hoverY = -1;
	}

	function clicked(i: number, j: number) {
		const coords: string = `${i.toString()}|${j.toString()}`;
		if (selectedNotes.has(coords)) selectedNotes.delete(coords);
		else selectedNotes.add(coords);
		console.log(selectedNotes);
		console.log(selectedNotes.has(coords));
	}

	function clearSelection() {
		selectedNotes = new Set();
	}

	// onMount(() => {
	// 	console.log(notes);
	// });
</script>

<div class="relative flex h-72 w-full">
	<div class="mr-4 flex h-full items-center justify-center">
		<TrebleClef />
	</div>
	<!-- <div class="relative mr-8 h-full w-32"> -->
	<!-- 	<TimeSignature /> -->
	<!-- </div> -->

	<div class="flex h-full w-full -translate-y-8">
		<!-- Horizontal Lines -->
		{#each notes as note, i}
			<div class="flex w-full flex-col">
				<!-- Vertical Lines -->
				{#each new Array(15) as _, j}
					<div
						role="button"
						tabindex="0"
						onkeydown={() => {}}
						class="w-[{columnWidth}%] relative h-5"
						onmouseenter={() => mouseEnterSheet(i, j)}
						onmouseleave={() => mouseLeaveSheet()}
						onclick={() => clicked(i, j)}
					>
						<div
							class="absolute top-[50%] w-full border-white
                    {j % 2 == 0 ? 'border-b' : ''} 
                    {j == 0 || j == 2 || j == 14 ? 'border-opacity-25' : ''}"
						></div>
						{#if (i == hoverX && j == hoverY) || selectedNotes.has(`${i.toString()}|${j.toString()}`)}
							<Quarter y={j} />
						{/if}
					</div>
				{/each}
			</div>
		{/each}
	</div>
</div>
