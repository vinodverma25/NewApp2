
// Canvas Taint Protection for Video Effects
class CanvasTaintProtection {
    constructor() {
        this.cleanCanvas = null;
        this.ctx = null;
        this.initialize();
    }

    initialize() {
        // Create a clean canvas for processing
        this.cleanCanvas = document.createElement('canvas');
        this.ctx = this.cleanCanvas.getContext('2d');
        
        // Set canvas to avoid taint issues
        this.ctx.imageSmoothingEnabled = true;
        this.ctx.imageSmoothingQuality = 'medium';
    }

    // Extract frame data safely from video element
    extractVideoFrame(videoElement, width = 320, height = 180) {
        try {
            // Set canvas dimensions
            this.cleanCanvas.width = width;
            this.cleanCanvas.height = height;
            
            // Clear canvas first
            this.ctx.clearRect(0, 0, width, height);
            
            // Draw video frame to clean canvas
            this.ctx.drawImage(videoElement, 0, 0, width, height);
            
            // Extract data safely
            return this.cleanCanvas.toDataURL('image/jpeg', 0.8);
            
        } catch (error) {
            console.warn('Canvas taint detected, using fallback method:', error);
            return this.createFallbackFrame(width, height);
        }
    }

    // Create a fallback frame when video is tainted
    createFallbackFrame(width = 320, height = 180) {
        try {
            this.cleanCanvas.width = width;
            this.cleanCanvas.height = height;
            
            // Create a simple placeholder
            this.ctx.fillStyle = '#000000';
            this.ctx.fillRect(0, 0, width, height);
            
            // Add text
            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = '16px Arial';
            this.ctx.textAlign = 'center';
            this.ctx.fillText('Processing...', width / 2, height / 2);
            
            return this.cleanCanvas.toDataURL('image/jpeg', 0.8);
            
        } catch (error) {
            console.error('Failed to create fallback frame:', error);
            // Return minimal base64 image
            return 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/2wBDAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIRAxEAPwDX/9k=';
        }
    }

    // Process image data to remove taint
    cleanImageData(imageData) {
        try {
            // Create new clean canvas
            const tempCanvas = document.createElement('canvas');
            const tempCtx = tempCanvas.getContext('2d');
            
            // Create image from data
            const img = new Image();
            img.crossOrigin = 'anonymous'; // Prevent taint
            
            return new Promise((resolve, reject) => {
                img.onload = () => {
                    try {
                        tempCanvas.width = img.width;
                        tempCanvas.height = img.height;
                        
                        // Draw to clean canvas
                        tempCtx.drawImage(img, 0, 0);
                        
                        // Extract clean data
                        const cleanData = tempCanvas.toDataURL('image/jpeg', 0.8);
                        resolve(cleanData);
                        
                    } catch (error) {
                        reject(error);
                    }
                };
                
                img.onerror = reject;
                img.src = imageData;
            });
            
        } catch (error) {
            console.error('Failed to clean image data:', error);
            return Promise.resolve(this.createFallbackFrame());
        }
    }

    // Set CORS attributes on video elements
    setupVideoElement(videoElement) {
        try {
            videoElement.crossOrigin = 'anonymous';
            videoElement.setAttribute('crossorigin', 'anonymous');
            
            // Add load event listener to handle CORS issues
            videoElement.addEventListener('loadstart', () => {
                console.log('Video loading started with CORS protection');
            });
            
            videoElement.addEventListener('error', (e) => {
                console.warn('Video load error, possibly due to CORS:', e);
                // Try without CORS as fallback
                if (videoElement.crossOrigin) {
                    videoElement.crossOrigin = null;
                    videoElement.removeAttribute('crossorigin');
                    videoElement.load(); // Reload without CORS
                }
            });
            
        } catch (error) {
            console.warn('Failed to setup video CORS protection:', error);
        }
    }
}

// Global instance
window.canvasTaintProtection = new CanvasTaintProtection();

// Utility function to safely extract video frame
window.safeExtractVideoFrame = function(videoElement, width, height) {
    return window.canvasTaintProtection.extractVideoFrame(videoElement, width, height);
};

// Utility function to clean image data
window.cleanImageData = function(imageData) {
    return window.canvasTaintProtection.cleanImageData(imageData);
};

// Auto-setup video elements
document.addEventListener('DOMContentLoaded', function() {
    // Setup all video elements with CORS protection
    const videos = document.querySelectorAll('video');
    videos.forEach(video => {
        window.canvasTaintProtection.setupVideoElement(video);
    });
    
    // Monitor for new video elements
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    if (node.tagName === 'VIDEO') {
                        window.canvasTaintProtection.setupVideoElement(node);
                    } else if (node.querySelectorAll) {
                        const videos = node.querySelectorAll('video');
                        videos.forEach(video => {
                            window.canvasTaintProtection.setupVideoElement(video);
                        });
                    }
                }
            });
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
