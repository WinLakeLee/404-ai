// Main JavaScript for 404 AI Application

document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const fileName = document.getElementById('fileName');
    const preview = document.getElementById('preview');
    const previewImage = document.getElementById('previewImage');
    const results = document.getElementById('results');
    const resultContent = document.getElementById('resultContent');
    const loading = document.getElementById('loading');
    const submitBtn = document.getElementById('submitBtn');
    
    // File input change handler
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        
        if (file) {
            fileName.textContent = file.name;
            
            // Show preview
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                preview.classList.remove('hidden');
            };
            reader.readAsDataURL(file);
            
            // Hide previous results
            results.classList.add('hidden');
        }
    });
    
    // Form submit handler
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const file = fileInput.files[0];
        if (!file) {
            alert('Please select a file');
            return;
        }
        
        // Show loading, hide results
        loading.classList.remove('hidden');
        results.classList.add('hidden');
        submitBtn.disabled = true;
        
        // Create form data
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            // Hide loading
            loading.classList.add('hidden');
            submitBtn.disabled = false;
            
            if (data.success) {
                displayResults(data.result);
            } else {
                displayError(data.error || 'An error occurred');
            }
        } catch (error) {
            loading.classList.add('hidden');
            submitBtn.disabled = false;
            displayError('Network error: ' + error.message);
        }
    });
    
    function displayResults(result) {
        results.classList.remove('hidden');
        
        let html = '<div class="result-items">';
        
        // Status
        if (result.status) {
            html += `
                <div class="result-item">
                    <span class="result-label">Status:</span>
                    <span class="result-value">${result.status}</span>
                </div>
            `;
        }
        
        // Anomaly Detection
        if (result.anomaly_detected !== undefined) {
            const anomalyClass = result.anomaly_detected ? 'anomaly-detected' : 'no-anomaly';
            const anomalyText = result.anomaly_detected ? 'Anomaly Detected' : 'No Anomaly Detected';
            html += `
                <div class="result-item">
                    <span class="result-label">Detection:</span>
                    <span class="result-value ${anomalyClass}">${anomalyText}</span>
                </div>
            `;
        }
        
        // Anomaly Score
        if (result.anomaly_score !== undefined) {
            html += `
                <div class="result-item">
                    <span class="result-label">Anomaly Score:</span>
                    <span class="result-value">${result.anomaly_score.toFixed(4)}</span>
                </div>
            `;
        }
        
        // Confidence
        if (result.confidence !== undefined) {
            html += `
                <div class="result-item">
                    <span class="result-label">Confidence:</span>
                    <span class="result-value">${(result.confidence * 100).toFixed(2)}%</span>
                </div>
            `;
        }
        
        // Additional data
        if (result.raw_output) {
            html += `
                <div class="result-item">
                    <span class="result-label">Raw Output:</span>
                    <span class="result-value">${result.raw_output}</span>
                </div>
            `;
        }
        
        html += '</div>';
        
        resultContent.innerHTML = html;
    }
    
    function displayError(message) {
        results.classList.remove('hidden');
        resultContent.innerHTML = `
            <div style="color: #dc3545; padding: 20px; text-align: center;">
                <strong>Error:</strong> ${message}
            </div>
        `;
    }
    
    // Health check on page load
    checkHealth();
    
    async function checkHealth() {
        try {
            const response = await fetch('/health');
            const data = await response.json();
            
            console.log('Health check:', data);
            
            if (!data.model_loaded) {
                console.warn('Model not loaded');
            }
        } catch (error) {
            console.error('Health check failed:', error);
        }
    }
});
