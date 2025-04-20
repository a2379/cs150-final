<script lang="ts">
	import MusicBoard from '$lib/components/MusicBoard.svelte';
	const genres: Array<string> = ['Jazz', 'Gospel', 'Rock'];
	let bpm: number = $state(120);
	let selectedGenre: string = $state('Jazz');
	let gridState: Array<Array<number>> = $state([new Array(16).fill(0.0), new Array(16).fill(0.0)]);
</script>

<div class="h-screen w-screen overflow-hidden bg-[#121212] text-[#ebe8e8]">
	<form
		method="POST"
		action="?/generate"
		class="flex h-full w-full flex-col items-center justify-center overflow-hidden"
	>
		<!-- Top Menu -->
		<div class="flex h-[10%] w-full items-center justify-center space-x-20">
			<label class="input w-48 focus:border-transparent focus:ring-0 focus:outline-none">
				<span class="label">BPM</span>
				<input name="bpm" type="text" bind:value={bpm} />
			</label>

			<div class="join">
				{#each genres as genre}
					<input
						bind:group={selectedGenre}
						value={genre}
						class="join-item btn btn-dash btn-success"
						type="radio"
						name="genre"
						aria-label={genre}
					/>
				{/each}
			</div>

			<button class="btn btn-soft btn-success" type="submit">Generate</button>
		</div>

		<!-- Music Board -->
		<div class="flex h-[90%] w-full items-center justify-center">
			<MusicBoard bind:gridState />
		</div>
	</form>
</div>
