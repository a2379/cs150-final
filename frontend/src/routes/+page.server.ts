import type { Actions } from '@sveltejs/kit';

export const actions: Actions = {
  generate: async ({ request }) => {
    const data = await request.formData();
    const bpmData = data.get('bpm') as string;
    const genreData = data.get('genre') as string;
    const gridStateData = data.get('gridState') as string;

    const genres: string[] = ['jazz', 'gospel', 'rock'];

    if (
      (typeof bpmData === "string" && Number.isNaN(Number(bpmData))) ||
      !genres.includes(genreData.toLowerCase())
    )
      return { success: false, message: 'Invalid form data.' };

    const bpm = Number(bpmData);
    const genre = genreData.toLowerCase();
    const gridState: Array<Array<number>> = JSON.parse(gridStateData);

    const response = await fetch('http://127.0.0.1:5000/api/play', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        bpm: bpm,
        genre: genre,
        grid1: gridState[0],
        grid2: gridState[1],
      }),
    });

    if (response.ok)
      return { success: true, message: 'Music Generated!' };
    else
      return { success: false, message: 'Failed to Generate Music' };
  }
};

