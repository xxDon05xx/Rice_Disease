document.addEventListener('DOMContentLoaded', () => {
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewContainer = document.getElementById('preview-container');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const diseaseNameSpan = document.getElementById('disease-name');
    const confidenceScoreSpan = document.getElementById('confidence-score');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const recommendationBox = document.getElementById('recommendation-box');
    const recommendationText = document.getElementById('recommendation-text');
    const weatherBar = document.getElementById('weather-bar');
    const severityContainer = document.getElementById('severity-container');
    const severityScore = document.getElementById('severity-score');

    // Modal elements
    const locationModal = document.getElementById('location-modal');
    const allowLocationBtn = document.getElementById('allow-location-btn');
    const denyLocationBtn = document.getElementById('deny-location-btn');

    // Stores current weather data to send with prediction
    let weatherData = { temp_c: null, humidity: null, is_rain: false };

    // --- WEATHER + LOCATION ---
    function fetchWeather(lat, lon) {
        // Now strictly using provided coordinates to ensure permission was granted
        fetch(`https://wttr.in/${lat},${lon}?format=j1`)
            .then(res => res.json())
            .then(data => {
                const current = data.current_condition[0];
                const temp = current.temp_C;
                const humidity = current.humidity;
                const desc = current.weatherDesc[0].value;
                const rainMm = parseFloat(current.precipMM || 0);
                const isRain = rainMm > 0;

                // Store for prediction request
                weatherData = {
                    temp_c: parseFloat(temp),
                    humidity: parseFloat(humidity),
                    is_rain: isRain
                };

                // Pick weather icon
                const descLower = desc.toLowerCase();
                let icon = '🌤️';
                if (descLower.includes('rain') || descLower.includes('drizzle')) icon = '🌧️';
                else if (descLower.includes('thunder')) icon = '⛈️';
                else if (descLower.includes('cloud')) icon = '☁️';
                else if (descLower.includes('snow')) icon = '❄️';
                else if (descLower.includes('fog') || descLower.includes('mist')) icon = '🌫️';
                else if (descLower.includes('sunny') || descLower.includes('clear')) icon = '☀️';

                // Display weather badge
                weatherBar.innerHTML = `
                    <div class="weather-card">
                        <span class="weather-icon">${icon}</span>
                        <span class="weather-detail"><strong>${temp}°C</strong></span>
                        <span class="weather-sep">·</span>
                        <span class="weather-detail">💧 ${humidity}%</span>
                        <span class="weather-sep">·</span>
                        <span class="weather-detail weather-desc">${desc}</span>
                    </div>
                `;
                weatherBar.classList.remove('hidden');
            })
            .catch(err => {
                console.warn('Weather fetch failed:', err);
            });
    }

    function requestLocation() {
        if (!navigator.geolocation) return;

        // This triggers the browser permission prompt
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const { latitude, longitude } = pos.coords;
                fetchWeather(latitude, longitude);
            },
            (err) => {
                console.warn('Location permission denied or unavailable.', err.message);
                // Weather remains hidden if denied
            },
            { timeout: 10000 }
        );
    }

    // Check permission status instead of directly requesting
    if (navigator.permissions && navigator.permissions.query) {
        navigator.permissions.query({ name: 'geolocation' }).then(function (result) {
            if (result.state === 'granted') {
                requestLocation();
            } else if (result.state === 'prompt') {
                locationModal.classList.remove('hidden');
            }
            // If denied, we do nothing
        });
    } else {
        // Fallback for browsers that don't support permissions.query
        locationModal.classList.remove('hidden');
    }

    if (allowLocationBtn) {
        allowLocationBtn.addEventListener('click', function (e) {
            e.preventDefault();
            console.log("Allow clicked");
            locationModal.classList.add('hidden');
            requestLocation();
        });
    }

    if (denyLocationBtn) {
        denyLocationBtn.addEventListener('click', function (e) {
            e.preventDefault();
            console.log("Deny clicked");
            locationModal.classList.add('hidden');
        });
    }
    // --- FILE SELECTION ---
    fileInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                imagePreview.src = e.target.result;
                previewContainer.classList.remove('hidden');
                analyzeBtn.classList.remove('hidden');
                resultSection.classList.add('hidden');
            };
            reader.readAsDataURL(file);
        }
    });

    // --- PREDICTION ---
    analyzeBtn.addEventListener('click', () => {
        const file = fileInput.files[0];
        if (!file) return;

        resultSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultContent.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', file);

        // Append weather data if available
        if (weatherData.temp_c !== null) {
            formData.append('temp_c', weatherData.temp_c);
            formData.append('humidity', weatherData.humidity);
            formData.append('is_rain', weatherData.is_rain);
        }

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
            .then(response => response.json())
            .then(data => {
                loader.classList.add('hidden');
                resultContent.classList.remove('hidden');

                if (data.error) {
                    diseaseNameSpan.textContent = "തകരാർ: " + data.error;
                    confidenceScoreSpan.textContent = "-";
                    recommendationBox.classList.add('hidden');
                    if (severityContainer) severityContainer.classList.add('hidden');
                } else {
                    diseaseNameSpan.textContent = data.class;
                    confidenceScoreSpan.textContent = data.confidence.toFixed(2);

                    if (severityContainer && data.severity !== undefined) {
                        const sev = parseFloat(data.severity);
                        let sevColor = '#2ecc71';
                        if (sev >= 60) sevColor = '#c0392b';
                        else if (sev >= 35) sevColor = '#e74c3c';
                        else if (sev >= 15) sevColor = '#f39c12';

                        if (data.class === 'ആരോഗ്യമുള്ള ഇല') {
                            severityContainer.classList.add('hidden');
                        } else {
                            severityScore.textContent = `${sev}% (${data.severity_label})`;
                            severityScore.style.backgroundColor = sevColor;
                            severityContainer.classList.remove('hidden');
                        }
                    } else if (severityContainer) {
                        severityContainer.classList.add('hidden');
                    }

                    if (data.recommendation) {
                        recommendationText.textContent = data.recommendation;
                        recommendationBox.classList.remove('hidden');
                    } else {
                        recommendationBox.classList.add('hidden');
                    }
                }
            })
            .catch(error => {
                console.error('Error:', error);
                loader.classList.add('hidden');
                resultContent.classList.remove('hidden');
                diseaseNameSpan.textContent = "സെർവറുമായി ബന്ധപ്പെടാൻ സാധിക്കുന്നില്ല.";
                recommendationBox.classList.add('hidden');
            });
    });
});
