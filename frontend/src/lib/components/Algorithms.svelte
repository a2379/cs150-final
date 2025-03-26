<script lang='ts'>
  let selectedAlgorithm;;
</script>

<div class="h-full w-1/2 flex items-center justify-center">
  <select bind:value={selectedAlgorithm}
          class="select">
    <option value="Algorithm 1">Algorithm 1</option>
    <option value="Algorithm 2">Algorithm 2</option>
    <option value="Algorithm 3">Algorithm 3</option>
    <option value="Algorithm 4">Algorithm 4</option>
  </select>
</div>
