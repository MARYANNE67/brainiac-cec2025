const fileInput = document.querySelector("#file");
const preview = document.querySelector("#image-preview");
const title = document.querySelector(".dropzone__title");

if (fileInput && preview && title) {
    fileInput.addEventListener("change", () => {
        const [file] = fileInput.files;
        if (!file) return;

        preview.src = URL.createObjectURL(file);
        preview.onload = () => URL.revokeObjectURL(preview.src);
        title.textContent = file.name;
    });
}
