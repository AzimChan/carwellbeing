// Car Wellbeing Analyzer - Frontend Logic
class CarAnalyzer {
    constructor() {
        this.selectedFile = null;
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        // File upload elements
        this.dropZone = document.getElementById('dropZone');
        this.fileInput = document.getElementById('fileInput');
        this.imagePreview = document.getElementById('imagePreview');
        this.previewImg = document.getElementById('previewImg');
        this.fileName = document.getElementById('fileName');
        this.removeImage = document.getElementById('removeImage');
        this.analyzeBtn = document.getElementById('analyzeBtn');
        this.analyzeText = document.getElementById('analyzeText');
        this.analyzeLoading = document.getElementById('analyzeLoading');

        // Results elements
        this.overallStatus = document.getElementById('overallStatus');
        this.statusBadge = document.getElementById('statusBadge');
        this.tipsContent = document.getElementById('tipsContent');

        // AI Analysis elements
        this.aiCarLevel = document.getElementById('aiCarLevel');
        this.aiCarValue = document.getElementById('aiCarValue');
        this.aiDentLevel = document.getElementById('aiDentLevel');
        this.aiDentValue = document.getElementById('aiDentValue');
        this.aiRustLevel = document.getElementById('aiRustLevel');
        this.aiRustValue = document.getElementById('aiRustValue');
        this.aiScratchLevel = document.getElementById('aiScratchLevel');
        this.aiScratchValue = document.getElementById('aiScratchValue');
        this.aiStarRating = document.getElementById('aiStarRating');
        this.aiRatingWarning = document.getElementById('aiRatingWarning');
    }

    attachEventListeners() {
        // Drag and drop events
        this.dropZone.addEventListener('click', () => this.fileInput.click());
        this.dropZone.addEventListener('dragover', (e) => this.handleDragOver(e));
        this.dropZone.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        this.dropZone.addEventListener('drop', (e) => this.handleDrop(e));

        // File input change
        this.fileInput.addEventListener('change', (e) => this.handleFileSelect(e));

        // Remove image
        this.removeImage.addEventListener('click', () => this.removeSelectedImage());

        // Analyze button
        this.analyzeBtn.addEventListener('click', () => this.analyzeCar());
    }

    handleDragOver(e) {
        e.preventDefault();
        this.dropZone.classList.add('border-primary', 'bg-primary/5');
    }

    handleDragLeave(e) {
        e.preventDefault();
        this.dropZone.classList.remove('border-primary', 'bg-primary/5');
    }

    handleDrop(e) {
        e.preventDefault();
        this.dropZone.classList.remove('border-primary', 'bg-primary/5');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.selectFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.selectFile(file);
        }
    }

    selectFile(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            alert('File size must be less than 10MB.');
            return;
        }

        this.selectedFile = file;
        this.displayImagePreview(file);
        this.enableAnalyzeButton();
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.fileName.textContent = file.name;
            this.imagePreview.classList.remove('hidden');
            this.dropZone.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    removeSelectedImage() {
        this.selectedFile = null;
        this.imagePreview.classList.add('hidden');
        this.dropZone.classList.remove('hidden');
        this.fileInput.value = '';
        this.disableAnalyzeButton();
        this.resetResults();
    }

    enableAnalyzeButton() {
        this.analyzeBtn.disabled = false;
    }

    disableAnalyzeButton() {
        this.analyzeBtn.disabled = true;
    }

    async analyzeCar() {
        if (!this.selectedFile) return;

        this.showLoadingState();
        
        try {
            // Create FormData for file upload
            const formData = new FormData();
            formData.append('file', this.selectedFile);

            // Call actual FastAPI endpoint
            const response = await this.callAnalysisAPI(formData);
            
            // Display results
            this.displayResults(response);
            
        } catch (error) {
            console.error('Analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
        } finally {
            this.hideLoadingState();
        }
    }

    // Simulate analysis - replace with actual API call
    async simulateAnalysis() {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Generate random scores for demonstration
        const cleanness = Math.floor(Math.random() * 40) + 60; // 60-100
        const integrity = Math.floor(Math.random() * 30) + 70; // 70-100
        
        return {
            cleanness: {
                score: cleanness,
                description: this.getCleannessDescription(cleanness)
            },
            integrity: {
                score: integrity,
                description: this.getIntegrityDescription(integrity)
            },
            tips: this.generateTips(cleanness, integrity)
        };
    }

    // API call to your FastAPI backend
    async callAnalysisAPI(formData) {
        console.log('Sending request to API...');
        console.log('FormData contents:', formData);
        
        try {
            const response = await fetch('http://localhost:3000/api/analyze', {
                method: 'POST',
                body: formData
            });
            
            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);
            
            if (!response.ok) {
                let errorMessage = 'Analysis failed';
                try {
                    const errorData = await response.json();
                    console.log('Error response:', errorData);
                    errorMessage = errorData.detail || errorData.message || 'Analysis failed';
                } catch (e) {
                    console.log('Could not parse error response as JSON');
                    errorMessage = `HTTP ${response.status}: ${response.statusText}`;
                }
                throw new Error(errorMessage);
            }
            
            const result = await response.json();
            console.log('Success response:', result);
            return result;
            
        } catch (error) {
            console.error('API call failed:', error);
            throw error;
        }
    }

    getCleannessDescription(score) {
        if (score >= 90) return 'Excellent! Your car is spotless and well-maintained.';
        if (score >= 80) return 'Very good condition with minor cleaning needed.';
        if (score >= 70) return 'Good condition, but could use a thorough cleaning.';
        if (score >= 60) return 'Fair condition, significant cleaning recommended.';
        return 'Poor condition, immediate cleaning required.';
    }

    getIntegrityDescription(score) {
        if (score >= 90) return 'Excellent structural integrity, no visible damage.';
        if (score >= 80) return 'Very good condition with minor cosmetic issues.';
        if (score >= 70) return 'Good condition, some wear visible but functional.';
        if (score >= 60) return 'Fair condition, some damage present.';
        return 'Poor condition, significant damage detected.';
    }

    generateTips(cleanness, integrity) {
        const tips = [];
        
        if (cleanness < 80) {
            tips.push('Consider a professional car wash and interior detailing');
            tips.push('Regular washing prevents paint damage and maintains value');
        }
        
        if (integrity < 80) {
            tips.push('Schedule a professional inspection for any visible damage');
            tips.push('Address minor issues before they become major problems');
        }
        
        if (cleanness >= 80 && integrity >= 80) {
            tips.push('Great job maintaining your car! Keep up the regular maintenance');
            tips.push('Consider protective coatings to maintain current condition');
        }
        
        return tips.length > 0 ? tips : ['Your car is in excellent condition!'];
    }

    displayResults(results) {
        // Update AI damage assessment with percentages
        this.updateAIDamageAssessment(results);
        
        // Update overall status
        this.updateOverallStatus(results);
        
        // Update tips
        this.updateTips(results.tips);
        
        // Show results section
        this.overallStatus.classList.remove('hidden');
    }

    updateAIDamageAssessment(results) {
        // Reset all values
        this.aiCarLevel.value = 0;
        this.aiCarValue.textContent = '0%';
        this.aiDentLevel.value = 0;
        this.aiDentValue.textContent = '0%';
        this.aiRustLevel.value = 0;
        this.aiRustValue.textContent = '0%';
        this.aiScratchLevel.value = 0;
        this.aiScratchValue.textContent = '0%';

        // Display damage percentages if available
        if (results.damage_analysis && results.damage_analysis.all_probabilities) {
            const probs = results.damage_analysis.all_probabilities;
            
            // Update car detection
            const carPercent = Math.round(probs.car * 100);
            this.aiCarLevel.value = carPercent;
            this.aiCarValue.textContent = `${carPercent}%`;
            
            // Update dent detection
            const dentPercent = Math.round(probs.dent * 100);
            this.aiDentLevel.value = dentPercent;
            this.aiDentValue.textContent = `${dentPercent}%`;
            
            // Update rust detection
            const rustPercent = Math.round(probs.rust * 100);
            this.aiRustLevel.value = rustPercent;
            this.aiRustValue.textContent = `${rustPercent}%`;
            
            // Update scratch detection
            const scratchPercent = Math.round(probs.scratch * 100);
            this.aiScratchLevel.value = scratchPercent;
            this.aiScratchValue.textContent = `${scratchPercent}%`;
            
            // Update AI star rating
            this.updateAIStarRating(probs);
        }
    }

    updateAIStarRating(probs) {
        const carProb = probs.car || 0;
        const damageProb = Math.max(probs.dent || 0, probs.rust || 0, probs.scratch || 0);
        
        // Calculate star rating based on car detection and damage levels
        let starCount;
        if (carProb > 0.8 && damageProb < 0.3) {
            starCount = 5; // Excellent
        } else if (carProb > 0.6 && damageProb < 0.5) {
            starCount = 4; // Very Good
        } else if (carProb > 0.4 && damageProb < 0.7) {
            starCount = 3; // Good
        } else if (damageProb < 0.8) {
            starCount = 2; // Fair
        } else {
            starCount = 1; // Needs Attention
        }
        
        this.displayAIStarRating(starCount);
        this.checkForAIRepairWarning(starCount, probs);
    }

    displayAIStarRating(starCount) {
        // Clear existing stars
        this.aiStarRating.innerHTML = '';

        // Create star SVG
        const starSVG = `
            <svg class="star" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z"/>
            </svg>
        `;

        // Add stars
        for (let i = 0; i < 5; i++) {
            const starElement = document.createElement('div');
            starElement.innerHTML = starSVG;
            const star = starElement.querySelector('.star');
            
            if (i < starCount) {
                star.classList.add('filled');
                
                // Apply color based on star count
                if (starCount <= 2) {
                    star.classList.add('red');
                } else if (starCount === 3) {
                    star.classList.add('yellow');
                }
                // 4-5 stars remain golden (default filled color)
            }
            
            this.aiStarRating.appendChild(starElement);
        }

        // Add rating text
        const ratingText = document.createElement('span');
        ratingText.className = 'ml-2 text-sm font-medium text-muted-foreground';
        ratingText.textContent = `${starCount}/5 stars`;
        this.aiStarRating.appendChild(ratingText);
    }

    checkForAIRepairWarning(starCount, probs) {
        // Clear any existing classes
        this.aiRatingWarning.className = 'hidden text-center p-3 rounded-md';
        
        if (starCount <= 2) {
            // Red - Not permitted to drive
            const issues = [];
            if ((probs.dent || 0) > 0.5) issues.push('dents');
            if ((probs.rust || 0) > 0.5) issues.push('rust');
            if ((probs.scratch || 0) > 0.5) issues.push('scratches');
            
            this.aiRatingWarning.innerHTML = `🚫 Not permitted to drive: ${issues.join(', ')}`;
            this.aiRatingWarning.classList.add('bg-red-100', 'border', 'border-red-300', 'text-red-800');
            this.aiRatingWarning.classList.remove('hidden');
        } else if (starCount === 3) {
            // Yellow - Okay to drive
            this.aiRatingWarning.innerHTML = `⚠️ Okay to drive - consider repairs soon`;
            this.aiRatingWarning.classList.add('bg-yellow-100', 'border', 'border-yellow-300', 'text-yellow-800');
            this.aiRatingWarning.classList.remove('hidden');
        } else if (starCount >= 4) {
            // Green - Awesome condition
            this.aiRatingWarning.innerHTML = `✅ Awesome condition - safe to drive!`;
            this.aiRatingWarning.classList.add('bg-green-100', 'border', 'border-green-300', 'text-green-800');
            this.aiRatingWarning.classList.remove('hidden');
        }
    }

    updateOverallStatus(results) {
        // Calculate overall status based on damage probabilities
        let status, statusClass;
        
        if (results.damage_analysis && results.damage_analysis.all_probabilities) {
            const probs = results.damage_analysis.all_probabilities;
            const carProb = probs.car || 0;
            const damageProb = Math.max(probs.dent || 0, probs.rust || 0, probs.scratch || 0);
            
            if (carProb > 0.8 && damageProb < 0.3) {
                status = 'Excellent';
                statusClass = 'bg-green-100 text-green-800';
            } else if (carProb > 0.6 && damageProb < 0.5) {
                status = 'Very Good';
                statusClass = 'bg-blue-100 text-blue-800';
            } else if (carProb > 0.4 && damageProb < 0.7) {
                status = 'Good';
                statusClass = 'bg-yellow-100 text-yellow-800';
            } else if (damageProb < 0.8) {
                status = 'Fair';
                statusClass = 'bg-orange-100 text-orange-800';
            } else {
                status = 'Needs Attention';
                statusClass = 'bg-red-100 text-red-800';
            }
        } else {
            status = 'Analysis Complete';
            statusClass = 'bg-gray-100 text-gray-800';
        }
        
        this.statusBadge.textContent = status;
        this.statusBadge.className = `inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${statusClass}`;
    }

    updateTips(tips) {
        this.tipsContent.innerHTML = tips.map(tip => 
            `<div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                <p class="text-sm text-muted-foreground">${tip}</p>
            </div>`
        ).join('');
    }

    resetResults() {
        // Reset AI analysis elements
        this.aiCarLevel.value = 0;
        this.aiCarValue.textContent = '0%';
        this.aiDentLevel.value = 0;
        this.aiDentValue.textContent = '0%';
        this.aiRustLevel.value = 0;
        this.aiRustValue.textContent = '0%';
        this.aiScratchLevel.value = 0;
        this.aiScratchValue.textContent = '0%';
        
        // Clear AI star rating
        this.aiStarRating.innerHTML = '';
        this.aiRatingWarning.classList.add('hidden');
        
        this.overallStatus.classList.add('hidden');
        
        this.tipsContent.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                <p class="text-sm text-muted-foreground">Upload a car image to get damage analysis and maintenance recommendations</p>
            </div>
        `;
    }

    showLoadingState() {
        this.analyzeBtn.disabled = true;
        this.analyzeText.classList.add('hidden');
        this.analyzeLoading.classList.remove('hidden');
    }

    hideLoadingState() {
        this.analyzeBtn.disabled = false;
        this.analyzeText.classList.remove('hidden');
        this.analyzeLoading.classList.add('hidden');
    }

    showError(message) {
        alert(message); // Replace with better error handling UI
    }


}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const analyzer = new CarAnalyzer();
});
