<script lang="ts">
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Separator } from '$lib/components/ui/separator';
	import MusicSheet from '$lib/components/MusicSheet.svelte';

	const algorithms: Array<string> = [
		'Neural Network',
		'Riffology',
		'Cellular Automata',
		'Monte Carlo',
		'Serialism'
	];

	let selectedIndex: number = $state(-1);
</script>

<div class="flex h-screen w-screen flex-col items-center justify-center">
	<div class="flex h-1/2 w-full items-center justify-center">
		<!-- Algorithms Selection -->
		<div class="flex w-1/3 flex-col items-center justify-center">
			<div class="mb-4 text-sm font-medium leading-none">Algorithms</div>
			<ScrollArea class="w-1/2 rounded-md border">
				{#each algorithms as alg, i}
					<button
						onclick={() => (selectedIndex = i)}
						class="w-full p-4 text-start text-sm hover:bg-gray-600
                  {i == selectedIndex ? 'bg-gray-600' : ''}"
					>
						{alg}
					</button>
					<Separator />
				{/each}
			</ScrollArea>
		</div>

		<!-- Music Sheet -->
		<div class="mr-12 flex h-1/2 w-2/3 items-center justify-center">
			<MusicSheet />
		</div>

		<!-- <div class="flex w-1/2 items-center justify-center"> -->
		<!-- 	<div class="w-72"> -->
		<!-- 		<Slider bind:value type="single" max={100} step={1} /> -->
		<!-- 	</div> -->
		<!-- </div> -->
	</div>
	<div class="flex h-1/2 w-full items-center justify-center">
		<div class="text-3xl">[Display Result Here]</div>
	</div>
</div>
