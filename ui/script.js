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
        this.cleannessScore = document.getElementById('cleannessScore');
        this.cleannessBar = document.getElementById('cleannessBar');
        this.cleannessDescription = document.getElementById('cleannessDescription');
        this.integrityScore = document.getElementById('integrityScore');
        this.integrityBar = document.getElementById('integrityBar');
        this.integrityDescription = document.getElementById('integrityDescription');
        this.overallStatus = document.getElementById('overallStatus');
        this.statusBadge = document.getElementById('statusBadge');
        this.tipsContent = document.getElementById('tipsContent');
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
            formData.append('image', this.selectedFile);

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
        const response = await fetch('http://localhost:8000/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }
        
        return await response.json();
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
        // Update cleanness results
        this.cleannessScore.textContent = `${results.cleanness.score}%`;
        this.cleannessBar.style.width = `${results.cleanness.score}%`;
        this.cleannessDescription.textContent = results.cleanness.description;

        // Update integrity results
        this.integrityScore.textContent = `${results.integrity.score}%`;
        this.integrityBar.style.width = `${results.integrity.score}%`;
        this.integrityDescription.textContent = results.integrity.description;

        // Update overall status
        this.updateOverallStatus(results);
        
        // Update tips
        this.updateTips(results.tips);
        
        // Show results section
        this.overallStatus.classList.remove('hidden');
    }

    updateOverallStatus(results) {
        const avgScore = (results.cleanness.score + results.integrity.score) / 2;
        let status, statusClass;
        
        if (avgScore >= 90) {
            status = 'Excellent';
            statusClass = 'bg-green-100 text-green-800';
        } else if (avgScore >= 80) {
            status = 'Very Good';
            statusClass = 'bg-blue-100 text-blue-800';
        } else if (avgScore >= 70) {
            status = 'Good';
            statusClass = 'bg-yellow-100 text-yellow-800';
        } else if (avgScore >= 60) {
            status = 'Fair';
            statusClass = 'bg-orange-100 text-orange-800';
        } else {
            status = 'Needs Attention';
            statusClass = 'bg-red-100 text-red-800';
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
        this.cleannessScore.textContent = '--';
        this.cleannessBar.style.width = '0%';
        this.cleannessDescription.textContent = 'Upload an image to see cleanness analysis';
        
        this.integrityScore.textContent = '--';
        this.integrityBar.style.width = '0%';
        this.integrityDescription.textContent = 'Upload an image to see integrity analysis';
        
        this.overallStatus.classList.add('hidden');
        
        this.tipsContent.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                <p class="text-sm text-muted-foreground">Upload a car image to get personalized maintenance recommendations</p>
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
    new CarAnalyzer();
});
