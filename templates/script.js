const API_URL = "http://localhost:8080/predict";

function predict() {
    const input = document.getElementById("imageInput");
    const resultDiv = document.getElementById("result");
    const previewImage = document.getElementById("previewImage");

    if (!input.files[0]) {
        alert("Please select an image first!");
        return;
    }

    const file = input.files[0];

    // Preview image
    const reader = new FileReader();
    reader.onload = () => {
        previewImage.src = reader.result;
        previewImage.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Prepare form data
    const formData = new FormData();
    formData.append("image", file);

    resultDiv.innerHTML = "⏳ Predicting...";

    fetch(API_URL, {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            resultDiv.innerHTML = "❌ Error: " + data.error;
        } else {
            resultDiv.innerHTML = `
                🩺 Prediction: <span style="color: green">${data.prediction}</span><br>
                📊 Confidence: ${(data.confidence * 100).toFixed(2)}%
            `;
        }
    })
    .catch(error => {
        resultDiv.innerHTML = "❌ Failed to connect to server";
        console.error(error);
    });
}
