function validateForm() 
{
  const fileInput = document.querySelector('input[type="file"]');
  if (!fileInput.value) {
    alert("Please select a fingerprint file to upload.");
    return false;
  }
  return true;
}
