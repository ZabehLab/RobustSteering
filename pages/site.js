"use strict";

const dialog = document.querySelector("#figure-dialog");
const dialogImage = document.querySelector("#dialog-image");
const dialogCaption = document.querySelector("#dialog-caption");

document.querySelectorAll("[data-figure]").forEach((button) => {
  button.addEventListener("click", () => {
    if (typeof dialog.showModal !== "function") {
      window.open(button.dataset.figure, "_blank", "noopener");
      return;
    }
    dialogImage.src = button.dataset.figure;
    dialogImage.alt = button.querySelector("img").alt;
    dialogCaption.textContent = button.dataset.caption;
    dialog.showModal();
  });
});

dialog.querySelector(".dialog-close").addEventListener("click", () => dialog.close());
dialog.addEventListener("click", (event) => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right ||
      event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});

const copyButton = document.querySelector("#copy-citation");
const copyStatus = document.querySelector("#copy-status");
copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector("#bibtex").textContent.trim());
    copyButton.textContent = "Copied";
    copyStatus.textContent = "BibTeX copied to clipboard.";
    window.setTimeout(() => { copyButton.textContent = "Copy BibTeX"; }, 2000);
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector("#bibtex"));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = "BibTeX selected. Use your device’s copy command.";
  }
});
