document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const movieId = document.getElementById('movieId').value;
    const userId = document.getElementById('userId').value;
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                movie_id: movieId,
                user_id: userId
            })
        });
        
        const data = await response.json();
        
        const resultDiv = document.getElementById('predictionResult');
        const resultContent = document.getElementById('resultContent');
        
        if (data.success) {
            resultContent.innerHTML = `
                <p><strong>Movie name:</strong> ${data.movie_name}</p>
                <p><strong>User name:</strong> ${data.user_name}</p>
                <p><strong>Predicted Rating:</strong> ${data.prediction.toFixed(2)}</p>
            `;
        } else {
            resultContent.innerHTML = `
                <div class="alert alert-danger">
                    Error: ${data.error}
                </div>
            `;
        }
        
        resultDiv.style.display = 'block';
    } catch (error) {
        console.error('Error:', error);
        const resultDiv = document.getElementById('predictionResult');
        const resultContent = document.getElementById('resultContent');
        resultContent.innerHTML = `
            <div class="alert alert-danger">
                An error occurred while making the prediction.
            </div>
        `;
        resultDiv.style.display = 'block';
    }
}); 