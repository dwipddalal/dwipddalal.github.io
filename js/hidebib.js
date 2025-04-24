// Function to toggle the visibility of a bibliography block
function toggleblock(blockId) {
  var block = document.getElementById(blockId);
  if (block) {
    if (block.style.display == 'none') {
      block.style.display = 'block';
    } else {
      block.style.display = 'none';
    }
  }
}

// Function to hide a block by default
function hideblock(blockId) {
  var block = document.getElementById(blockId);
  if (block) {
    block.style.display = 'none';
  }
} 