#!/usr/bin/env python3
import sys
import os
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QWidget, QPushButton, QFileDialog, QMessageBox,
                             QStatusBar, QProgressBar, QDialog, QHBoxLayout, 
                             QGroupBox, QRadioButton, QButtonGroup, QSpinBox,
                             QCheckBox, QDialogButtonBox, QScrollArea, QFrame)
from PySide6.QtCore import Qt, QMimeData, QSize, QRect, Signal, QPoint
from PySide6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage, QPainter, QPen, QColor

import numpy as np
from PIL import Image
from cli import PixelArtDownscaler

class ImageViewer(QLabel):
    """Base class for image viewers with zoom and pan capabilities"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 1px solid #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 5px;
            }
        """)
        self.setMinimumSize(500, 500)
        
        # Image viewing properties
        self.image = None
        self.pixmap = None
        self.zoom_factor = 8  # Default zoom level
        
        # Panning properties
        self.is_panning = False
        self.pan_start_pos = None
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.space_pressed = False  # Track spacebar state
        
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)  # Allow keyboard focus
    
    def setImage(self, image):
        """Set the image from a PIL Image object"""
        self.image = image
        self.updatePixmap()
        
    def updatePixmap(self):
        """Update the pixmap from the current image"""
        if not self.image:
            return
            
        # Convert PIL image to QImage
        img_data = self.image.convert("RGBA").tobytes("raw", "RGBA")
        q_img = QImage(img_data, self.image.width, self.image.height, self.image.width * 4, QImage.Format_RGBA8888)
        
        # Create pixmap
        self.pixmap = QPixmap.fromImage(q_img)
        
        # Clear cached zoom data
        if hasattr(self, 'cached_pixmap'):
            del self.cached_pixmap
        if hasattr(self, 'cached_zoom'):
            del self.cached_zoom
            
        # Clear text
        self.setText("")
        
        # Force update
        self.update()
    
    def paintEvent(self, event):
        super().paintEvent(event)  # Call parent's paintEvent to maintain label functionality
        
        if not self.pixmap:
            return
        
        # Cap zoom to prevent rendering issues
        MAX_ZOOM = 10  # Maximum zoom factor for better performance
        if self.zoom_factor > MAX_ZOOM:
            self.zoom_factor = MAX_ZOOM
            
        # Ensure we have consistent zoom values as integers
        self.zoom_factor = int(self.zoom_factor)
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)  # No antialiasing for pixel art
        painter.setRenderHint(QPainter.SmoothPixmapTransform, False)  # Ensure crisp pixels
        
        try:
            # Basic integer scaling for pixel art
            pixmap_width = self.pixmap.width()
            pixmap_height = self.pixmap.height()
            
            # Calculate scaled dimensions - explicitly convert to integers
            scaled_width = int(pixmap_width * self.zoom_factor)
            scaled_height = int(pixmap_height * self.zoom_factor)
            
            # Calculate image position, center the image by default, then apply pan offset
            # Use integer division to ensure pixel-perfect positioning
            x_offset = (self.width() - scaled_width) // 2 + self.pan_offset_x
            y_offset = (self.height() - scaled_height) // 2 + self.pan_offset_y
            
            # Cache scaled pixmaps for better performance
            # We'll use a class attribute to store the cached scaled pixmap
            if not hasattr(self, 'cached_pixmap') or not hasattr(self, 'cached_zoom') or self.cached_zoom != self.zoom_factor:
                self.cached_pixmap = self.pixmap.scaled(
                    scaled_width, 
                    scaled_height,
                    Qt.IgnoreAspectRatio,  # We're doing precise scaling so keep exact dimensions
                    Qt.FastTransformation  # Use FastTransformation for nearest-neighbor
                )
                self.cached_zoom = self.zoom_factor
            
            # Draw the cached pixmap at the calculated position
            painter.drawPixmap(x_offset, y_offset, self.cached_pixmap)
            
            # Only draw grid when zoom factor is sufficient but not too large
            # Skip grid at high zoom to improve performance
            if 4 <= self.zoom_factor <= 8:
                # Use a semi-transparent grid color
                painter.setPen(QPen(QColor(200, 200, 200, 100), 1))
                
                # For improved performance, only draw visible grid lines
                # Determine visible area of the image
                visible_start_x = max(0, int((-x_offset) / self.zoom_factor))
                visible_end_x = min(self.pixmap.width(), 
                                   int((-x_offset + self.width()) / self.zoom_factor) + 1)
                
                visible_start_y = max(0, int((-y_offset) / self.zoom_factor))
                visible_end_y = min(self.pixmap.height(), 
                                   int((-y_offset + self.height()) / self.zoom_factor) + 1)
                
                # Draw only visible grid lines
                for x in range(visible_start_x, visible_end_x + 1):
                    grid_x = int(x_offset + x * self.zoom_factor)
                    if 0 <= grid_x <= self.width():
                        painter.drawLine(
                            grid_x, max(0, y_offset),
                            grid_x, min(self.height(), y_offset + scaled_height)
                        )
                
                for y in range(visible_start_y, visible_end_y + 1):
                    grid_y = int(y_offset + y * self.zoom_factor)
                    if 0 <= grid_y <= self.height():
                        painter.drawLine(
                            max(0, x_offset), grid_y,
                            min(self.width(), x_offset + scaled_width), grid_y
                        )
        except Exception as e:
            # Fallback in case of rendering error
            print(f"Error during painting: {str(e)}")
            painter.fillRect(self.rect(), QColor(240, 240, 240))
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.rect(), Qt.AlignCenter, f"Error rendering image: {str(e)}")
    
    def wheelEvent(self, event):
        # Zoom in/out with mouse wheel
        delta = event.angleDelta().y()
        old_zoom = self.zoom_factor
        
        # Calculate new zoom factor
        if delta > 0:
            # Zoom in (with bounds check) - limit to practical maximum
            self.zoom_factor = min(10, self.zoom_factor + 1)
        elif delta < 0:
            # Zoom out (with bounds check)
            self.zoom_factor = max(1, self.zoom_factor - 1)
            
        # Only update if the zoom actually changed
        if old_zoom != self.zoom_factor:
            # Update this view
            self.update()
            
            # Sync zoom across viewers if in a main window
            if hasattr(self.main_window, 'syncZoom'):
                self.main_window.syncZoom(self.zoom_factor, source_viewer=self)
    
    def keyPressEvent(self, event):
        # Track spacebar state for panning
        if event.key() == Qt.Key_Space:
            self.space_pressed = True
            # Change cursor to a hand when spacebar is pressed
            self.setCursor(Qt.OpenHandCursor)
    
    def keyReleaseEvent(self, event):
        # Reset spacebar state
        if event.key() == Qt.Key_Space:
            self.space_pressed = False
            # Reset cursor when spacebar is released
            self.setCursor(Qt.ArrowCursor)
            # Also reset panning state
            self.is_panning = False
    
    def mousePressEvent(self, event):
        # Pan with spacebar+left-drag or right-drag
        if (event.button() == Qt.LeftButton and self.space_pressed) or event.button() == Qt.RightButton:
            self.is_panning = True
            self.pan_start_pos = event.position()
            # Change cursor to closed hand during drag
            self.setCursor(Qt.ClosedHandCursor)
        
    def mouseMoveEvent(self, event):
        # Handle panning - both spacebar+drag and right-click+drag
        if self.is_panning:
            # Only proceed if left button is down during spacebar or right button is down
            if ((event.buttons() & Qt.LeftButton and self.space_pressed) or 
                (event.buttons() & Qt.RightButton)):
                
                # Calculate how much we've moved the mouse
                delta_x = event.position().x() - self.pan_start_pos.x()
                delta_y = event.position().y() - self.pan_start_pos.y()
                
                # Update the pan offset (with safety limits)
                self.pan_offset_x = max(-10000, min(10000, self.pan_offset_x + delta_x))
                self.pan_offset_y = max(-10000, min(10000, self.pan_offset_y + delta_y))
                
                # Update the start position for the next move
                self.pan_start_pos = event.position()
                
                # Redraw with the new pan offset
                self.update()
                
                # Sync pan across viewers if in a main window
                if hasattr(self.main_window, 'syncPan'):
                    self.main_window.syncPan(self.pan_offset_x, self.pan_offset_y, source_viewer=self)
    
    def mouseReleaseEvent(self, event):
        # Handle the end of panning
        if self.is_panning and (event.button() == Qt.LeftButton or event.button() == Qt.RightButton):
            self.is_panning = False
            
            # Change cursor depending on which button was released
            if event.button() == Qt.RightButton:
                self.setCursor(Qt.ArrowCursor)
            elif self.space_pressed:
                # For spacebar+left button, keep open hand cursor while spacebar is down
                self.setCursor(Qt.OpenHandCursor)


class DropArea(ImageViewer):
    pixelSelected = Signal(int, int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setText("Drop your pixel art image here\nor click to select an image")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 20px;
                font-size: 16px;
            }
        """)
        self.setAcceptDrops(True)
        
        # Selection properties
        self.dragging = False
        self.selection_rect = QRect()
        self.selection_start = None
        self.pixel_size = 1
        self.offset_x = 0
        self.offset_y = 0
        
    def setImage(self, image_path):
        self.image = Image.open(image_path)
        super().updatePixmap()  # Use parent's updatePixmap
    
    def paintEvent(self, event):
        super().paintEvent(event)  # Call parent's paintEvent
        
        if not self.pixmap:
            return
            
        painter = QPainter(self)
        
        # Draw selection rectangle if we have one
        if self.selection_rect and not self.selection_rect.isEmpty():
            scaled_width = self.pixmap.width() * self.zoom_factor
            scaled_height = self.pixmap.height() * self.zoom_factor
            
            # Recalculate offset with pan
            x_offset = int(max(-10000, min(10000, (self.width() - scaled_width) // 2 + self.pan_offset_x)))
            y_offset = int(max(-10000, min(10000, (self.height() - scaled_height) // 2 + self.pan_offset_y)))
            
            # Calculate screen coordinates for selection rectangle
            screen_rect = QRect(
                x_offset + self.selection_rect.x() * self.zoom_factor,
                y_offset + self.selection_rect.y() * self.zoom_factor,
                self.selection_rect.width() * self.zoom_factor,
                self.selection_rect.height() * self.zoom_factor
            )
            
            # Draw with highlight color
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(screen_rect)
    
    def mousePressEvent(self, event):
        # If no image is loaded, handle file selection
        if not self.pixmap:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.gif *.bmp)"
            )
            if file_path:
                self.main_window.loadImage(file_path)
            return
        
        # Handle panning with spacebar+left-click or right-click (use parent implementation)
        if (self.space_pressed and event.button() == Qt.LeftButton) or event.button() == Qt.RightButton:
            super().mousePressEvent(event)
            return
                
        # For pixel selection with left button (no spacebar)
        if event.button() == Qt.LeftButton:
            # Calculate image coordinates
            scaled_width = self.pixmap.width() * self.zoom_factor
            scaled_height = self.pixmap.height() * self.zoom_factor
            
            # Calculate offset with pan
            x_offset = int(max(-10000, min(10000, (self.width() - scaled_width) // 2 + self.pan_offset_x)))
            y_offset = int(max(-10000, min(10000, (self.height() - scaled_height) // 2 + self.pan_offset_y)))
            
            # Convert screen coordinates to image coordinates
            image_x = int((event.position().x() - x_offset) // self.zoom_factor)
            image_y = int((event.position().y() - y_offset) // self.zoom_factor)
            
            # Check if click is within image bounds
            if (0 <= image_x < self.pixmap.width() and 
                0 <= image_y < self.pixmap.height()):
                self.dragging = True
                self.selection_start = QPoint(image_x, image_y)
                self.selection_rect = QRect(image_x, image_y, 1, 1)
                self.update()
    
    def mouseMoveEvent(self, event):
        # Handle panning via parent class
        if self.is_panning:
            super().mouseMoveEvent(event)
            return
            
        # Handle selection dragging with left button (not during panning)
        if self.dragging and (event.buttons() & Qt.LeftButton) and self.pixmap and self.selection_start:
            # Calculate image coordinates
            scaled_width = self.pixmap.width() * self.zoom_factor
            scaled_height = self.pixmap.height() * self.zoom_factor
            
            # Calculate offset with pan
            x_offset = int(max(-10000, min(10000, (self.width() - scaled_width) // 2 + self.pan_offset_x)))
            y_offset = int(max(-10000, min(10000, (self.height() - scaled_height) // 2 + self.pan_offset_y)))
            
            # Convert screen coordinates to image coordinates
            image_x = max(0, min(self.pixmap.width() - 1, int((event.position().x() - x_offset) // self.zoom_factor)))
            image_y = max(0, min(self.pixmap.height() - 1, int((event.position().y() - y_offset) // self.zoom_factor)))
            
            # Update selection rectangle
            self.selection_rect = QRect(
                min(self.selection_start.x(), image_x),
                min(self.selection_start.y(), image_y),
                abs(image_x - self.selection_start.x()) + 1,
                abs(image_y - self.selection_start.y()) + 1
            )
            
            self.update()
    
    def mouseReleaseEvent(self, event):
        # Handle panning release via parent class
        if self.is_panning:
            super().mouseReleaseEvent(event)
            return
            
        # Handle selection release
        if not self.dragging or event.button() != Qt.LeftButton or not self.pixmap or not self.selection_start:
            return
            
        self.dragging = False
        
        # Calculate the pixel size and offset from the selection
        if not self.selection_rect.isEmpty() and self.selection_rect.width() > 0 and self.selection_rect.height() > 0:
            # The selected rectangle represents a single pixel in the original art
            self.pixel_size = max(1, self.selection_rect.width())  # Use width as the pixel size
            self.offset_x = self.selection_rect.x() % self.pixel_size  # Calculate offset
            self.offset_y = self.selection_rect.y() % self.pixel_size
            
            # Emit signal with selected pixel information
            self.pixelSelected.emit(self.pixel_size, self.offset_x, self.offset_y)
            
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("""
                QLabel {
                    border: 2px dashed #2980b9;
                    border-radius: 5px;
                    background-color: #e8f4fc;
                    padding: 20px;
                    font-size: 16px;
                }
            """)
        
    def dragLeaveEvent(self, event):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 20px;
                font-size: 16px;
            }
        """)
        
    def dropEvent(self, event: QDropEvent):
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                background-color: #f8f8f8;
                padding: 20px;
                font-size: 16px;
            }
        """)
        
        urls = event.mimeData().urls()
        if urls and len(urls) > 0:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                self.main_window.loadImage(file_path)
            else:
                QMessageBox.warning(self, "Invalid File", 
                                    "Please drop a valid image file (PNG, JPG, GIF, BMP).")

class PixelArtDownscalerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.downscaler = PixelArtDownscaler()
        self.current_image_path = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle("Pixel Art Scaler")
        self.setMinimumSize(1200, 800)
        
        # Central widget with horizontal split
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left side - Image view and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image view area
        image_group = QGroupBox("Original Image (Select a Pixel)")
        image_layout = QVBoxLayout(image_group)
        
        # Instructions for pixel selection
        pixel_instructions = QLabel("Draw a rectangle around ONE pixel in the original art to set the pixel size.")
        pixel_instructions.setWordWrap(True)
        image_layout.addWidget(pixel_instructions)
        
        # Drop area / image view
        self.drop_area = DropArea(self)
        image_layout.addWidget(self.drop_area)
        
        # Browse button
        self.browse_button = QPushButton("Browse for Image")
        self.browse_button.clicked.connect(self.browseImage)
        image_layout.addWidget(self.browse_button)
        
        # Zoom and pan controls
        controls_layout = QVBoxLayout()
        
        # Zoom controls
        zoom_layout = QHBoxLayout()
        zoom_in_button = QPushButton("+")
        zoom_out_button = QPushButton("-")
        self.zoom_label = QLabel("Zoom: 8x")
        
        zoom_in_button.clicked.connect(self.zoom_in)
        zoom_out_button.clicked.connect(self.zoom_out)
        
        zoom_layout.addWidget(QLabel("Zoom:"))
        zoom_layout.addWidget(zoom_out_button)
        zoom_layout.addWidget(self.zoom_label)
        zoom_layout.addWidget(zoom_in_button)
        zoom_layout.addStretch(1)
        
        controls_layout.addLayout(zoom_layout)
        
        # Pan instructions
        pan_instructions = QLabel("<i>Tip: Right-click and drag to pan the image, or hold SPACEBAR and drag</i>")
        pan_instructions.setAlignment(Qt.AlignCenter)
        controls_layout.addWidget(pan_instructions)
        
        image_layout.addLayout(controls_layout)
        
        # Current selection info
        self.pixel_info_label = QLabel("No pixel selected yet. Please drag a rectangle around a single pixel.")
        self.pixel_info_label.setWordWrap(True)
        image_layout.addWidget(self.pixel_info_label)
        
        left_layout.addWidget(image_group)
        
        # Right side - Configuration panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Explanation text
        explanation = QLabel(
            "<b>What this app does:</b><br>"
            "<i>Restores crisp pixel art from blurry or scaled versions:</i><br>"
            "1. Manually draw a box around a single pixel (the box should exactly outline one pixel in the original art)<br>"
            "2. The app detects the pixel size and grid offset based on your selection<br>"
            "3. Removes compression artifacts by downscaling to a true 1:1 pixel ratio<br>"
            "4. Re-upscales using nearest neighbor for perfect crisp pixels"
        )
        explanation.setWordWrap(True)
        explanation.setAlignment(Qt.AlignLeft)
        right_layout.addWidget(explanation)
        
        # Manual settings group
        manual_group = QGroupBox("Pixel Size & Offset Settings")
        manual_layout = QVBoxLayout(manual_group)
        
        manual_note = QLabel("<b>Pixel selection:</b> In the image preview, draw a rectangle that precisely covers ONE pixel in the original art. " 
                              "The app will calculate pixel size and offset from your selection. Alternatively, set these values manually:")
        manual_note.setWordWrap(True)
        manual_layout.addWidget(manual_note)
        
        # Scale input
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Pixel Size:"))
        self.scale_input = QSpinBox()
        self.scale_input.setRange(1, 32)
        self.scale_input.setValue(1)  # Default to 1 until detection
        # Removed automatic preview update
        scale_layout.addWidget(self.scale_input)
        manual_layout.addLayout(scale_layout)
        
        # Offset inputs
        offset_layout = QHBoxLayout()
        offset_layout.addWidget(QLabel("X Offset:"))
        self.offset_x_input = QSpinBox()
        self.offset_x_input.setRange(0, 31)
        self.offset_x_input.setValue(0)  # Default to 0
        offset_layout.addWidget(self.offset_x_input)
        
        offset_layout.addWidget(QLabel("Y Offset:"))
        self.offset_y_input = QSpinBox()
        self.offset_y_input.setRange(0, 31)
        self.offset_y_input.setValue(0)  # Default to 0
        offset_layout.addWidget(self.offset_y_input)
        manual_layout.addLayout(offset_layout)
        
        right_layout.addWidget(manual_group)
        
        # Color processing options group
        color_group = QGroupBox("Color Processing Options")
        color_layout = QVBoxLayout(color_group)
        
        # Color clustering options
        from PySide6.QtWidgets import QSlider
        color_threshold_layout = QHBoxLayout()
        color_threshold_label = QLabel("Color similarity threshold:")
        self.color_threshold_slider = QSlider(Qt.Horizontal)
        self.color_threshold_slider.setMinimum(0)
        self.color_threshold_slider.setMaximum(50)
        self.color_threshold_slider.setValue(15)  # Default value
        self.color_threshold_value = QLabel("15")
        
        self.color_threshold_slider.valueChanged.connect(lambda value: self.color_threshold_value.setText(str(value)))
        
        color_threshold_layout.addWidget(color_threshold_label)
        color_threshold_layout.addWidget(self.color_threshold_slider)
        color_threshold_layout.addWidget(self.color_threshold_value)
        color_layout.addLayout(color_threshold_layout)
        
        # Median option
        self.use_median_check = QCheckBox("Use median color (better for JPG compression)")
        self.use_median_check.setChecked(True)  # Enable by default for better performance and stability
        color_layout.addWidget(self.use_median_check)
        
        # Update preview button
        update_preview_button = QPushButton("Update Preview")
        update_preview_button.clicked.connect(self.updatePreview)
        color_layout.addWidget(update_preview_button)
        
        right_layout.addWidget(color_group)
        
        # Upscaling options group
        upscale_group = QGroupBox("Upscaling Options")
        upscale_layout = QVBoxLayout(upscale_group)
        
        # Original size upscale
        self.orig_size_check = QCheckBox("Create clean version at original size")
        self.orig_size_check.setChecked(True)
        upscale_layout.addWidget(self.orig_size_check)
        
        # Custom upscale
        custom_upscale_layout = QHBoxLayout()
        self.custom_upscale_check = QCheckBox("Create additional upscaled version: ")
        self.custom_upscale = QSpinBox()
        self.custom_upscale.setRange(2, 32)
        self.custom_upscale.setValue(8)
        self.custom_upscale.setEnabled(False)
        self.custom_upscale_check.toggled.connect(lambda checked: self.custom_upscale.setEnabled(checked))
        custom_upscale_layout.addWidget(self.custom_upscale_check)
        custom_upscale_layout.addWidget(self.custom_upscale)
        upscale_layout.addLayout(custom_upscale_layout)
        
        right_layout.addWidget(upscale_group)
        
        # Process button
        self.process_button = QPushButton("Process Image")
        self.process_button.setEnabled(False)  # Disabled until image is loaded
        self.process_button.clicked.connect(self.processLoadedImage)
        self.process_button.setStyleSheet("font-size: 14px; padding: 8px;")
        right_layout.addWidget(self.process_button)
        
        # Add spacer to push everything up
        right_layout.addStretch(1)
        
        # Preview section
        preview_group = QGroupBox("Downscaled Preview (True 1:1 Pixel Ratio)")
        preview_layout = QVBoxLayout(preview_group)
        
        # Preview description
        preview_description = QLabel("This shows how the image will look after being downscaled to true 1:1 pixel ratio. " +
                                   "This cleans up compression artifacts and prepares the image for crisp upscaling.")
        preview_description.setWordWrap(True)
        preview_layout.addWidget(preview_description)
        
        # Preview with instructions
        self.preview_viewer = ImageViewer(self)  # Use our custom ImageViewer class
        self.preview_viewer.setText("Select a pixel above and click 'Update Preview'")
        preview_layout.addWidget(self.preview_viewer)
        
        left_layout.addWidget(preview_group)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 7)  # 70% width
        main_layout.addWidget(right_panel, 3)  # 30% width
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        
        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def syncZoom(self, zoom_factor, source_viewer=None):
        """Synchronize zoom across all image viewers"""
        # Ensure zoom_factor is within valid range
        zoom_factor = min(10, max(1, zoom_factor))
        
        # Update zoom label
        self.zoom_label.setText(f"Zoom: {zoom_factor}x")
        
        # Get scale factor (with safety check)
        scale_to_use = max(1, self.scale_input.value())
        
        # Add flag to prevent recursive updates
        if hasattr(self, '_syncing_zoom') and self._syncing_zoom:
            return
            
        self._syncing_zoom = True
        
        try:
            # Update appropriate zoom levels based on source
            if source_viewer == self.preview_viewer and scale_to_use > 1:
                # If zooming from preview, update original
                if hasattr(self, 'drop_area'):
                    # Careful with integer division - use float division then convert to int
                    self.drop_area.zoom_factor = max(1, min(10, int(zoom_factor / scale_to_use)))
                    self.drop_area.update()
            else:
                # If zooming from original, update both
                if hasattr(self, 'drop_area'):
                    self.drop_area.zoom_factor = zoom_factor
                    self.drop_area.update()
                    
                # Update preview with scaled zoom
                if hasattr(self, 'preview_viewer') and scale_to_use > 1:
                    self.preview_viewer.zoom_factor = min(10, max(1, int(zoom_factor * scale_to_use)))
                    self.preview_viewer.update()
        finally:
            # Always clear the flag
            self._syncing_zoom = False
    
    def syncPan(self, pan_x, pan_y, source_viewer=None):
        """
        Synchronize panning across all image viewers
        
        Parameters:
            pan_x, pan_y: The pan offsets from the source viewer
            source_viewer: The viewer that initiated the pan (optional)
        """
        # Limit pan values to prevent integer overflow
        pan_x = max(-10000, min(10000, pan_x))
        pan_y = max(-10000, min(10000, pan_y))
        
        # Get the current scale factor
        scale_to_use = self.scale_input.value()
        
        # Apply to both viewers, maintaining their visual alignment
        if hasattr(self, 'drop_area'):
            self.drop_area.pan_offset_x = pan_x
            self.drop_area.pan_offset_y = pan_y
            self.drop_area.update()
            
        # Apply same pan to preview viewer (no scaling needed)
        if hasattr(self, 'preview_viewer'):
            self.preview_viewer.pan_offset_x = pan_x
            self.preview_viewer.pan_offset_y = pan_y
            self.preview_viewer.update()
    
    def zoom_in(self):
        if self.drop_area.zoom_factor < 10:  # Cap at 10x for better performance
            self.drop_area.zoom_factor += 1
            self.syncZoom(self.drop_area.zoom_factor)
    
    def zoom_out(self):
        if self.drop_area.zoom_factor > 1:
            self.drop_area.zoom_factor -= 1
            self.syncZoom(self.drop_area.zoom_factor)
    
    def onColorThresholdChanged(self, value):
        # Update the value label
        self.color_threshold_value.setText(str(value))
        # Update the preview
        self.updatePreview()
    
    def calculateFittingZoom(self, img_width, img_height, display_width, display_height):
        """Calculate zoom factor to fit an image within the display area"""
        # Apply some margin 
        display_width *= 0.9
        display_height *= 0.9
        
        # Calculate the zoom that fits within the display
        width_ratio = display_width / img_width
        height_ratio = display_height / img_height
        
        # Take the smaller ratio to fit within bounds
        fitting_zoom = min(width_ratio, height_ratio)
        
        # Round to an integer and constrain to reasonable range
        return max(1, min(32, round(fitting_zoom)))
    
    def updatePreview(self):
        # Only update if we have an image and pixel size is set
        if not self.current_image_path or self.scale_input.value() <= 1:
            return
        
        # Show loading state
        self.preview_viewer.setText("Processing preview...")
        QApplication.processEvents()
            
        try:
            # Get the current settings
            scale_to_use = self.scale_input.value()
            offset_x = self.offset_x_input.value()
            offset_y = self.offset_y_input.value()
            color_threshold = self.color_threshold_slider.value()
            use_median = self.use_median_check.isChecked()
            
            # Load the image
            img = Image.open(self.current_image_path)
            width, height = img.size
            
            # Apply offset if needed
            if offset_x > 0 or offset_y > 0:
                img = img.crop((offset_x, offset_y, width, height))
            
            print(f"DEBUG: Creating preview with scale={scale_to_use}, color_threshold={color_threshold}, use_median={use_median}")
            
            # Safety check for small images
            if width // scale_to_use < 1 or height // scale_to_use < 1:
                self.preview_viewer.setText("Image too small for selected scale")
                return
            
            # Create a downscaled preview using the current settings
            try:
                # Add a safety check - if median is not selected, ensure color clustering works
                if not use_median:
                    # If color clustering is likely to fail (with very small scale), use simple mean instead
                    if scale_to_use <= 2:
                        # For very small scales, force use_median to True as a safer alternative
                        use_median = True
                        print("DEBUG: Small scale detected, using median for safety")
                
                preview_img = self.downscaler.downscale_image(
                    img, 
                    scale_to_use, 
                    color_threshold=color_threshold, 
                    use_median=use_median
                )
                
                # Update the status
                self.status_bar.showMessage(f"Preview created at {preview_img.width}x{preview_img.height} pixels")
                
                # Display the preview in the viewer
                self.preview_viewer.setImage(preview_img)
                
                # Set the preview zoom factor to match the visual size of pixels
                # For a scale of N, we need an Nx zoom to match pixel sizes
                # This is because each pixel in the preview represents N pixels in the original
                orig_zoom = self.drop_area.zoom_factor
                
                # Apply scaled zoom to preview viewer to compensate for downscaling
                self.preview_viewer.zoom_factor = min(10, max(1, int(orig_zoom * scale_to_use)))
                
                # Maintain the existing pan position for the original
                orig_pan_x = self.drop_area.pan_offset_x
                orig_pan_y = self.drop_area.pan_offset_y
                
                # Scale the pan for the preview
                self.preview_viewer.pan_offset_x = int(orig_pan_x / scale_to_use)
                self.preview_viewer.pan_offset_y = int(orig_pan_y / scale_to_use)
                
                # Update both views
                self.drop_area.update()
                self.preview_viewer.update()
                
            except Exception as inner_e:
                print(f"DEBUG: Error in downscale_image: {str(inner_e)}")
                self.preview_viewer.setText(f"Error creating preview: {str(inner_e)}")
                return
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview update error: {str(e)}")
            print(f"DEBUG: Preview error: {str(e)}")
            self.preview_viewer.setText(f"Error updating preview: {str(e)}")
    
    def onPixelSelected(self, pixel_size, offset_x, offset_y):
        # Update UI with selected pixel information
        self.scale_input.setValue(pixel_size)
        self.offset_x_input.setValue(offset_x)
        self.offset_y_input.setValue(offset_y)
        
        # Update info label
        self.pixel_info_label.setText(
            f"Selected pixel size: {pixel_size}x{pixel_size} pixels\n"
            f"Grid offset: ({offset_x}, {offset_y}) pixels"
        )
    
    def browseImage(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.gif *.bmp)"
        )
        if file_path:
            self.loadImage(file_path)
    
    def loadImage(self, file_path):
        try:
            self.status_bar.showMessage(f"Loading: {os.path.basename(file_path)}")
            
            # Store the current image path
            self.current_image_path = file_path
            
            # Load the image for the pixel selection view
            self.drop_area.setImage(file_path)
            
            # Reset zoom and pan for both views
            self.drop_area.zoom_factor = 2  # Set input image to 2x by default
            self.drop_area.pan_offset_x = 0
            self.drop_area.pan_offset_y = 0
            self.preview_viewer.zoom_factor = 2
            self.preview_viewer.pan_offset_x = 0
            self.preview_viewer.pan_offset_y = 0
            # Update zoom label
            self.zoom_label.setText(f"Zoom: {self.drop_area.zoom_factor}x")
            
            # Update the zoom label
            self.zoom_label.setText(f"Zoom: {self.drop_area.zoom_factor}x")
            
            # Connect pixel selection signal if not already connected
            try:
                self.drop_area.pixelSelected.disconnect()
            except:
                pass
            self.drop_area.pixelSelected.connect(self.onPixelSelected)
            
            # Enable the process button
            self.process_button.setEnabled(True)
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}. Select a pixel or adjust settings manually.")
            
            # Create a simple test image for the preview area
            self.showTestPreview()
            
        except Exception as e:
            self.status_bar.showMessage("Error loading image")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def showTestPreview(self):
        """Show a test image in the preview to verify display works"""
        try:
            # Create a simple 3x3 test image
            test_img = Image.new("RGBA", (3, 3), (255, 255, 255, 255))
            
            # Add some colored pixels
            test_img.putpixel((0, 0), (255, 0, 0, 255))  # Red
            test_img.putpixel((1, 1), (0, 255, 0, 255))  # Green
            test_img.putpixel((2, 2), (0, 0, 255, 255))  # Blue
            
            # Scale up for better visibility
            scale = 50
            test_img = test_img.resize((3 * scale, 3 * scale), Image.NEAREST)
            
            # Update preview viewer
            self.preview_viewer.setImage(test_img)
            
            self.status_bar.showMessage("Test pattern displayed in preview")
            
        except Exception as e:
            print(f"Test preview error: {str(e)}")
            self.preview_viewer.setText(f"Error creating test preview: {str(e)}")
    
    def processLoadedImage(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
            
        try:
            file_path = self.current_image_path
            self.status_bar.showMessage(f"Processing: {os.path.basename(file_path)}")
            self.progress_bar.setVisible(True)
            QApplication.processEvents()
            
            # Load the image with PIL for preview
            pil_img = Image.open(file_path)
            width, height = pil_img.size
            
            # Get the scale and offset to use from the UI
            scale_to_use = self.scale_input.value()
            offset_x = self.offset_x_input.value()
            offset_y = self.offset_y_input.value()
            
            # Get other options
            export_original_size = self.orig_size_check.isChecked()
            color_threshold = self.color_threshold_slider.value()
            use_median = self.use_median_check.isChecked()
            custom_upscale_factor = None
            if self.custom_upscale_check.isChecked():
                custom_upscale_factor = self.custom_upscale.value()
            
            self.status_bar.showMessage(f"Using pixel size: {scale_to_use}x with offset ({offset_x}, {offset_y})")
            QApplication.processEvents()
            
            print(f"DEBUG: Processing with pixel size: {scale_to_use}x, offset: ({offset_x}, {offset_y})")
            
            # Safety check - if not using median, make sure it's safe
            if not use_median:
                # If color clustering is likely to fail (with very small scale), use simple mean instead
                if scale_to_use <= 2:
                    # For very small scales, force use_median to True as a safer alternative
                    use_median = True
                    print("DEBUG: Small scale detected, using median for safety")
                    # Inform the user that we're using median instead
                    self.status_bar.showMessage("Using median color for small scale (safer for small scales)")
                    QApplication.processEvents()
            
            # First crop the image if there's an offset
            if offset_x > 0 or offset_y > 0:
                # Crop to align with the grid
                pil_img = pil_img.crop((offset_x, offset_y, width, height))
                print(f"Image cropped to align with pixel grid using offset ({offset_x}, {offset_y})")
                # Create a temporary path for the cropped image
                temp_dir = os.path.dirname(file_path)
                temp_file = os.path.join(temp_dir, f"temp_cropped_{os.path.basename(file_path)}")
                pil_img.save(temp_file)
                file_path = temp_file
            
            downscaled_path, clean_path = self.downscaler.process_image(
                file_path, 
                force_scale=scale_to_use,
                upscale_factor=None,  # Will be auto-calculated in the function
                export_original_size=export_original_size,
                color_threshold=color_threshold,
                use_median=use_median
            )
            
            # Process additional custom upscale if requested
            if custom_upscale_factor and custom_upscale_factor > 1:
                self.status_bar.showMessage(f"Creating {custom_upscale_factor}x upscaled version...")
                QApplication.processEvents()
                
                custom_downscaled, custom_clean = self.downscaler.process_image(
                    file_path,
                    force_scale=scale_to_use,  # Use the same scale as main processing
                    upscale_factor=custom_upscale_factor,
                    export_original_size=False,
                    color_threshold=color_threshold,
                    use_median=use_median
                )
            
            # Create success message
            output_files = []
            if downscaled_path:
                output_files.append(f"• 1:1 pixel ratio: {os.path.basename(downscaled_path)}")
            if clean_path:
                output_files.append(f"• Clean upscaled: {os.path.basename(clean_path)}")
            if custom_upscale_factor and custom_clean:
                output_files.append(f"• {custom_upscale_factor}x upscaled: {os.path.basename(custom_clean)}")
                
            success_msg = "Processing complete!\n\nFiles created:\n" + "\n".join(output_files)
            
            # Show success message
            self.status_bar.showMessage("Processing complete!")
            QMessageBox.information(self, "Success", success_msg)
            
            # Show the result
            if clean_path:
                try:
                    # Load the clean upscaled image
                    clean_img = Image.open(clean_path)
                    
                    # Update the preview with the final result
                    self.preview_viewer.setImage(clean_img)
                    
                    # Set zoom level to compensate for downscaling
                    scale_to_use = self.scale_input.value()
                    self.preview_viewer.zoom_factor = min(10, max(1, int(self.drop_area.zoom_factor * scale_to_use)))
                    self.preview_viewer.update()
                    
                    # Update the status
                    self.status_bar.showMessage(f"Final clean version displayed in preview")
                except Exception as view_err:
                    print(f"Error displaying clean result: {str(view_err)}")
            
            # Clean up temporary file if it was created
            if offset_x > 0 or offset_y > 0:
                try:
                    os.remove(file_path)
                except:
                    pass
                    
        except Exception as e:
            self.status_bar.showMessage("Error")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
        
        finally:
            self.progress_bar.setVisible(False)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    window = PixelArtDownscalerApp()
    window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()