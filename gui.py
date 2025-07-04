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
        
        # Allow any positive zoom factor, including fractional values
        # Make sure it's not too small to see anything
        self.zoom_factor = max(0.1, self.zoom_factor)
            
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
            
            # Ensure we have at least 1 pixel width/height
            scaled_width = max(1, scaled_width)
            scaled_height = max(1, scaled_height)
            
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
            
            # Grid overlay has been removed to improve performance
        except Exception as e:
            # Fallback in case of rendering error
            painter.fillRect(self.rect(), QColor(240, 240, 240))
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(self.rect(), Qt.AlignCenter, f"Error rendering image: {str(e)}")
    
    def wheelEvent(self, event):
        # Zoom in/out with mouse wheel centered on cursor position
        delta = event.angleDelta().y()
        old_zoom = self.zoom_factor
        
        # Get mouse position
        mouse_pos = event.position()
        
        # Calculate image coordinates before zoom
        if self.pixmap:
            # Calculate image position and dimensions at current zoom
            pixmap_width = self.pixmap.width()
            pixmap_height = self.pixmap.height()
            scaled_width = pixmap_width * old_zoom
            scaled_height = pixmap_height * old_zoom
            x_offset = (self.width() - scaled_width) / 2 + self.pan_offset_x
            y_offset = (self.height() - scaled_height) / 2 + self.pan_offset_y
            
            # Calculate cursor position relative to image (in image coordinates)
            image_x = (mouse_pos.x() - x_offset) / old_zoom
            image_y = (mouse_pos.y() - y_offset) / old_zoom
            
            # Calculate new zoom factor
            if delta > 0:
                # Zoom in with smoother scaling
                self.zoom_factor = self.zoom_factor * 1.25  # Smoother zooming
            elif delta < 0:
                # Zoom out with smoother scaling
                self.zoom_factor = self.zoom_factor / 1.25  # Smoother zooming
            
            # Only update if the zoom actually changed
            if old_zoom != self.zoom_factor:
                # Calculate new scaled dimensions and offsets
                new_scaled_width = pixmap_width * self.zoom_factor
                new_scaled_height = pixmap_height * self.zoom_factor
                
                # Calculate new pan offsets to maintain cursor position over same image point
                new_x_offset = mouse_pos.x() - (image_x * self.zoom_factor)
                new_y_offset = mouse_pos.y() - (image_y * self.zoom_factor)
                
                # Calculate the adjustment needed from the center offset
                center_x_offset = (self.width() - new_scaled_width) / 2
                center_y_offset = (self.height() - new_scaled_height) / 2
                
                # Update pan offsets
                self.pan_offset_x = new_x_offset - center_x_offset
                self.pan_offset_y = new_y_offset - center_y_offset
                
                # Update this view
                self.update()
                
                # Sync zoom and pan across viewers if in a main window
                if hasattr(self.main_window, 'syncZoom'):
                    self.main_window.syncZoom(self.zoom_factor, source_viewer=self)
                    
                if hasattr(self.main_window, 'syncPan'):
                    self.main_window.syncPan(self.pan_offset_x, self.pan_offset_y, source_viewer=self)
    
    def keyPressEvent(self, event):
        # Track spacebar state for panning
        if event.key() == Qt.Key_Space:
            self.space_pressed = True
            # Change cursor to a hand when spacebar is pressed
            self.setCursor(Qt.OpenHandCursor)
        elif event.key() == Qt.Key_Shift:
            self.shift_pressed = True
            # If we're currently dragging, update the selection to be square
            if hasattr(self, 'dragging') and self.dragging:
                self.update()
    
    def keyReleaseEvent(self, event):
        # Reset spacebar state
        if event.key() == Qt.Key_Space:
            self.space_pressed = False
            # Reset cursor when spacebar is released
            self.setCursor(Qt.ArrowCursor)
            # Also reset panning state
            self.is_panning = False
        elif event.key() == Qt.Key_Shift:
            self.shift_pressed = False
            # If we're currently dragging, update the selection to reflect non-square
            if hasattr(self, 'dragging') and self.dragging:
                self.update()
    
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
    # Signal now includes width, height, offset_x, offset_y
    pixelSelected = Signal(int, int, int, int)
    
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
        self.pixel_width = 1
        self.pixel_height = 1
        self.offset_x = 0
        self.offset_y = 0
        self.shift_pressed = False  # Track shift key state
        
        # Grid overlay properties
        self.show_grid = True  # Show grid by default
        
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
            # Ensure pixel-perfect alignment by rounding to integers
            rect_x = round(x_offset + self.selection_rect.x() * self.zoom_factor)
            rect_y = round(y_offset + self.selection_rect.y() * self.zoom_factor)
            rect_width = round(self.selection_rect.width() * self.zoom_factor)
            rect_height = round(self.selection_rect.height() * self.zoom_factor)
            
            # Make sure the width and height are at least 1 pixel
            rect_width = max(1, rect_width)
            rect_height = max(1, rect_height)
            
            screen_rect = QRect(rect_x, rect_y, rect_width, rect_height)
            
            # Draw with highlight color
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.drawRect(screen_rect)
            
            # Draw grid overlay based on the number of pixels set in the main window
            # Only draw if grid visibility is enabled and there's a reasonable number of pixels
            if hasattr(self, 'show_grid') and self.show_grid and hasattr(self, 'main_window') and self.main_window is not None and hasattr(self.main_window, 'num_pixels_input'):
                num_pixels = self.main_window.num_pixels_input.value()
                if 1 < num_pixels <= 16:  # Only show grid for reasonable subdivisions (2-16)
                    # Draw grid lines to show the pixel subdivisions
                    painter.setPen(QPen(QColor(100, 100, 255, 180), 1))
                    
                    # Calculate the size of each division
                    division_width = rect_width / num_pixels
                    division_height = rect_height / num_pixels
                    
                    # Draw vertical grid lines
                    for i in range(1, num_pixels):
                        x = rect_x + i * division_width
                        painter.drawLine(int(x), rect_y, int(x), rect_y + rect_height)
                    
                    # Draw horizontal grid lines
                    for i in range(1, num_pixels):
                        y = rect_y + i * division_height
                        painter.drawLine(rect_x, int(y), rect_x + rect_width, int(y))
    
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
            # Ensure coordinates snap to pixel grid by using integer division, regardless of zoom factor
            image_x = int((event.position().x() - x_offset) / self.zoom_factor)
            image_y = int((event.position().y() - y_offset) / self.zoom_factor)
            
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
            # Clear cached selection rectangle when actively dragging
            if hasattr(self, '_cached_selection_rect'):
                delattr(self, '_cached_selection_rect')
            if hasattr(self, '_cached_selection_zoom'):
                delattr(self, '_cached_selection_zoom')
            if hasattr(self, '_cached_selection_pan'):
                delattr(self, '_cached_selection_pan')
            # Calculate image coordinates
            scaled_width = self.pixmap.width() * self.zoom_factor
            scaled_height = self.pixmap.height() * self.zoom_factor
            
            # Calculate offset with pan
            x_offset = int(max(-10000, min(10000, (self.width() - scaled_width) // 2 + self.pan_offset_x)))
            y_offset = int(max(-10000, min(10000, (self.height() - scaled_height) // 2 + self.pan_offset_y)))
            
            # Convert screen coordinates to image coordinates
            # Ensure coordinates snap to pixel grid with integer division, regardless of zoom factor
            image_x = max(0, min(self.pixmap.width() - 1, int((event.position().x() - x_offset) / self.zoom_factor)))
            image_y = max(0, min(self.pixmap.height() - 1, int((event.position().y() - y_offset) / self.zoom_factor)))
            
            # Determine which direction we're dragging
            drag_right = image_x >= self.selection_start.x()
            drag_down = image_y >= self.selection_start.y()
            
            # Get original end coordinates
            end_x = image_x
            end_y = image_y
            
            # If shift is pressed, make selection square
            if hasattr(self, 'shift_pressed') and self.shift_pressed:
                # Calculate the distance from start point in both directions
                dx = abs(end_x - self.selection_start.x())
                dy = abs(end_y - self.selection_start.y())
                
                # Use the smaller of the two sizes to make a square
                square_size = min(dx, dy)
                
                # Adjust end coordinates to create a square
                if drag_right:
                    end_x = self.selection_start.x() + square_size
                else:
                    end_x = self.selection_start.x() - square_size
                
                if drag_down:
                    end_y = self.selection_start.y() + square_size
                else:
                    end_y = self.selection_start.y() - square_size
            
            # Ensure coordinates are within image bounds
            end_x = max(0, min(self.pixmap.width() - 1, end_x))
            end_y = max(0, min(self.pixmap.height() - 1, end_y))
            
            # Create the selection rectangle
            self.selection_rect = QRect(
                min(self.selection_start.x(), end_x),
                min(self.selection_start.y(), end_y),
                abs(end_x - self.selection_start.x()) + 1,
                abs(end_y - self.selection_start.y()) + 1
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
            # Store both width and height for potentially non-square pixels
            self.pixel_width = max(1, self.selection_rect.width())
            self.pixel_height = max(1, self.selection_rect.height())
            
            # Calculate offsets based on the actual dimensions
            self.offset_x = self.selection_rect.x() % self.pixel_width
            self.offset_y = self.selection_rect.y() % self.pixel_height
            
            # Emit signal with pixel information - we'll include both width and height
            self.pixelSelected.emit(self.pixel_width, self.pixel_height, self.offset_x, self.offset_y)
            
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
                                    
    def resizeEvent(self, event):
        # Call parent implementation
        super().resizeEvent(event)
        
        # Clear the cached selection rectangle to force recalculation on resize
        if hasattr(self, '_cached_selection_rect'):
            delattr(self, '_cached_selection_rect')
        if hasattr(self, '_cached_selection_zoom'):
            delattr(self, '_cached_selection_zoom')
        if hasattr(self, '_cached_selection_pan'):
            delattr(self, '_cached_selection_pan')

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
        
        # Fractional scaling inputs
        # Note: QLabel is already imported at the top, so we only need QDoubleSpinBox and QFrame
        from PySide6.QtWidgets import QDoubleSpinBox, QFrame
        
        # Selection size and number of pixels selection
        selection_layout = QHBoxLayout()
        
        # Selection size (width/height of selection in pixels)
        selection_layout.addWidget(QLabel("Selection Size:"))
        self.selection_size_input = QSpinBox()
        self.selection_size_input.setRange(1, 1000)
        self.selection_size_input.setValue(1)  # Default to 1 until detection
        self.selection_size_input.valueChanged.connect(self.updateFractionalScale)
        selection_layout.addWidget(self.selection_size_input)
        selection_layout.addWidget(QLabel("pixels"))
        
        # Number of pixels in selection (for fractional scaling)
        selection_layout.addWidget(QLabel("Contains:"))
        self.num_pixels_input = QSpinBox()
        self.num_pixels_input.setRange(1, 100)
        self.num_pixels_input.setValue(1)  # Default to 1 (meaning exactly 1 pixel)
        self.num_pixels_input.valueChanged.connect(self.updateFractionalScale)
        selection_layout.addWidget(self.num_pixels_input)
        selection_layout.addWidget(QLabel("pixels"))
        
        manual_layout.addLayout(selection_layout)
        
        # Scale display (calculated from selection size and number of pixels)
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Calculated Scale:"))
        
        # Use QLabel instead of QSpinBox for displaying calculated scale
        self.scale_display = QLabel("1.00")
        self.scale_display.setStyleSheet("font-weight: bold; color: #2980b9;")
        self.scale_display.setMinimumWidth(60)
        scale_layout.addWidget(self.scale_display)
        
        # Add explanation label
        scale_explanation = QLabel("(Selection Size ÷ Num Pixels)")
        scale_explanation.setStyleSheet("font-style: italic; color: #666;")
        scale_layout.addWidget(scale_explanation)
        scale_layout.addStretch(1)
        
        manual_layout.addLayout(scale_layout)
        
        # Add checkbox for showing grid in selection
        grid_layout = QHBoxLayout()
        self.show_grid_check = QCheckBox("Show grid overlay in selection")
        self.show_grid_check.setChecked(True)  # Enable by default
        self.show_grid_check.setToolTip("Show grid lines in the selection rectangle to visualize pixel divisions")
        grid_layout.addWidget(self.show_grid_check)
        
        # Connect to toggle grid visibility
        self.show_grid_check.toggled.connect(self.toggleGridVisibility)
        
        # Set initial grid visibility on the drop area
        self.drop_area.show_grid = True
        
        manual_layout.addLayout(grid_layout)
        
        # Add a separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        manual_layout.addWidget(separator)
        
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
        
        # Simple color processing option (renamed from 'use median')
        self.use_median_check = QCheckBox("Simple color processing (faster, better for JPG compression)")
        self.use_median_check.setChecked(True)  # Enable by default for better performance and stability
        self.use_median_check.setToolTip("When enabled, uses median color for each pixel block. When disabled, uses advanced color clustering.")
        color_layout.addWidget(self.use_median_check)
        
        # Advanced color processing section (hidden by default)
        self.advanced_color_widget = QWidget()
        advanced_color_layout = QVBoxLayout(self.advanced_color_widget)
        advanced_color_layout.setContentsMargins(20, 5, 5, 5)  # Add left margin for nested appearance
        
        # Add description for advanced color processing
        advanced_description = QLabel("<i>Advanced color processing clusters similar colors together, like a limited color palette.</i>")
        advanced_description.setWordWrap(True)
        advanced_color_layout.addWidget(advanced_description)
        
        # Color clustering options
        from PySide6.QtWidgets import QSlider
        color_threshold_layout = QHBoxLayout()
        color_threshold_label = QLabel("Color similarity threshold:")
        self.color_threshold_slider = QSlider(Qt.Horizontal)
        self.color_threshold_slider.setMinimum(0)
        self.color_threshold_slider.setMaximum(50)
        self.color_threshold_slider.setValue(15)  # Default value
        self.color_threshold_value = QLabel("15")
        self.color_threshold_slider.setToolTip("Higher values combine more similar colors. Lower values preserve more distinct colors.")
        
        self.color_threshold_slider.valueChanged.connect(lambda value: self.color_threshold_value.setText(str(value)))
        
        color_threshold_layout.addWidget(color_threshold_label)
        color_threshold_layout.addWidget(self.color_threshold_slider)
        color_threshold_layout.addWidget(self.color_threshold_value)
        advanced_color_layout.addLayout(color_threshold_layout)
        
        # Add the advanced color widget to the main color layout
        color_layout.addWidget(self.advanced_color_widget)
        
        # Connect the checkbox to show/hide advanced options
        self.use_median_check.toggled.connect(lambda checked: self.advanced_color_widget.setVisible(not checked))
        
        # Initially hide advanced options since median is checked by default
        self.advanced_color_widget.setVisible(False)
        
        # Ignore outer pixels option
        ignore_outer_layout = QHBoxLayout()
        self.ignore_outer_check = QCheckBox("Ignore outer pixels:")
        self.ignore_outer_check.setChecked(True)  # Enable by default for better results
        ignore_outer_layout.addWidget(self.ignore_outer_check)
        
        # Slider for percentage of outer pixels to ignore
        self.ignore_outer_slider = QSlider(Qt.Horizontal)
        self.ignore_outer_slider.setMinimum(0)
        self.ignore_outer_slider.setMaximum(90)  # 0-90% range
        self.ignore_outer_slider.setValue(10)  # Default to 10%
        self.ignore_outer_value = QLabel("10%")
        
        # Connect slider to value label
        self.ignore_outer_slider.valueChanged.connect(lambda value: self.ignore_outer_value.setText(f"{value}%"))
        # Enable/disable slider based on checkbox
        self.ignore_outer_check.toggled.connect(self.ignore_outer_slider.setEnabled)
        
        ignore_outer_layout.addWidget(self.ignore_outer_slider)
        ignore_outer_layout.addWidget(self.ignore_outer_value)
        
        # Add tooltip
        self.ignore_outer_check.setToolTip("Ignore outer pixels when determining color to reduce edge artifacts")
        self.ignore_outer_slider.setToolTip("Percentage of the outer pixels to ignore (0-90%)")
        
        # Enable slider only if checkbox is checked
        self.ignore_outer_slider.setEnabled(self.ignore_outer_check.isChecked())
        
        color_layout.addLayout(ignore_outer_layout)
        
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
        
        # Preserve aspect ratio option
        self.preserve_aspect_check = QCheckBox("Preserve original pixel aspect ratio during upscaling (non-square pixels)")
        self.preserve_aspect_check.setChecked(True)  # Enable by default
        self.preserve_aspect_check.setToolTip("When checked, preserves the original rectangular pixels during upscaling. When unchecked, forces square pixels in the upscaled output.")
        upscale_layout.addWidget(self.preserve_aspect_check)
        
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
        # Allow any positive zoom factor, including fractional values
        # (no longer constrained to 1-10)
        zoom_factor = max(0.1, zoom_factor)
        
        # Calculate the scales based on selection size and number of pixels
        selection_size = self.selection_size_input.value() if hasattr(self, 'selection_size_input') else 1
        num_pixels = self.num_pixels_input.value() if hasattr(self, 'num_pixels_input') else 1
        
        # Get separate scales for X and Y if available
        if hasattr(self, 'scale_x') and hasattr(self, 'scale_y'):
            scale_x = self.scale_x
            scale_y = self.scale_y
            # Use average for the preview scaling
            scale_to_use = (scale_x + scale_y) / 2
        else:
            # Fall back to using selection_size if separate scales aren't set
            scale_to_use = selection_size / max(1, num_pixels)
        
        # Add flag to prevent recursive updates
        if hasattr(self, '_syncing_zoom') and self._syncing_zoom:
            return
            
        self._syncing_zoom = True
        
        try:
            # Synchronize zoom between views
            
            # Update zoom factors accounting for scaling between images
            if source_viewer == self.preview_viewer:
                # If zoom came from preview viewer, update drop area zoom accordingly
                if hasattr(self, 'drop_area'):
                    # Calculate appropriate zoom for the input image
                    # We need to account for the fact that if the scale is N, each preview pixel
                    # represents N input pixels, so the input zoom should be the preview zoom divided by N
                    orig_zoom = max(0.1, zoom_factor / scale_to_use)
                    self.drop_area.zoom_factor = orig_zoom
                    self.drop_area.update()
                    # Update zoom label to show the input image zoom level
                    self.zoom_label.setText(f"Zoom: {orig_zoom:.2f}x")
            else:
                # If zoom came from drop area or elsewhere, update the drop area first
                if hasattr(self, 'drop_area'):
                    self.drop_area.zoom_factor = zoom_factor
                    self.drop_area.update()
                    # Update zoom label to show the input image zoom level
                    self.zoom_label.setText(f"Zoom: {zoom_factor:.2f}x")
                
                # Then scale the preview zoom to match visual size
                if hasattr(self, 'preview_viewer'):
                    # Calculate appropriate zoom for the preview image
                    # If the scale is N, each preview pixel represents N input pixels
                    # so we need N times more zoom to maintain the same visual size
                    preview_zoom = max(0.1, zoom_factor * scale_to_use)
                    self.preview_viewer.zoom_factor = preview_zoom
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
        
        # Add flag to prevent recursive updates
        if hasattr(self, '_syncing_pan') and self._syncing_pan:
            return
            
        self._syncing_pan = True
        
        try:
            # Simply apply the exact same pan offset to both viewers
            # This approach ensures they move together at the same rate
            if source_viewer == self.drop_area:
                # Panning originated from input image
                if hasattr(self, 'preview_viewer'):
                    self.preview_viewer.pan_offset_x = pan_x
                    self.preview_viewer.pan_offset_y = pan_y
                    self.preview_viewer.update()
            elif source_viewer == self.preview_viewer:
                # Panning originated from preview image
                if hasattr(self, 'drop_area'):
                    self.drop_area.pan_offset_x = pan_x
                    self.drop_area.pan_offset_y = pan_y
                    self.drop_area.update()
            else:
                # Panning from somewhere else or initial setup
                if hasattr(self, 'drop_area'):
                    self.drop_area.pan_offset_x = pan_x
                    self.drop_area.pan_offset_y = pan_y
                    self.drop_area.update()
                
                if hasattr(self, 'preview_viewer'):
                    self.preview_viewer.pan_offset_x = pan_x
                    self.preview_viewer.pan_offset_y = pan_y
                    self.preview_viewer.update()
        finally:
            # Always clear the flag
            self._syncing_pan = False
    
    def zoom_in(self):
        if self.drop_area.zoom_factor < 10:  # Cap at 10x for better performance
            self.drop_area.zoom_factor += 1
            self.syncZoom(self.drop_area.zoom_factor)
    
    def zoom_out(self):
        if self.drop_area.zoom_factor > 1:
            self.drop_area.zoom_factor -= 1
            self.syncZoom(self.drop_area.zoom_factor)
    
    def updateFractionalScale(self):
        """Calculate and update the fractional scale based on user inputs"""
        selection_size = self.selection_size_input.value()
        num_pixels = self.num_pixels_input.value()
        
        if num_pixels > 0:
            # Get the pixel dimensions from the selection or use defaults if not available
            pixel_width = getattr(self.drop_area, 'pixel_width', selection_size)
            pixel_height = getattr(self.drop_area, 'pixel_height', selection_size)
            
            # Calculate scales for each dimension
            scale_x = pixel_width / num_pixels
            scale_y = pixel_height / num_pixels
            
            # Update the scale display - show both dimensions if they differ
            if abs(scale_x - scale_y) > 0.01:
                self.scale_display.setText(f"X:{scale_x:.2f}, Y:{scale_y:.2f}")
            else:
                self.scale_display.setText(f"{scale_x:.2f}")
            
            # Store the scales as properties of the main window
            self.scale_x = scale_x
            self.scale_y = scale_y
            
            # Force the drop area to redraw with the updated grid
            if hasattr(self, 'drop_area'):
                self.drop_area.update()
            
            # Get the top-left coordinates of the selection rectangle if we have one
            selection_x = 0
            selection_y = 0
            if hasattr(self.drop_area, 'selection_rect') and not self.drop_area.selection_rect.isEmpty():
                selection_x = self.drop_area.selection_rect.x()
                selection_y = self.drop_area.selection_rect.y()
            
            # Calculate the proper offset based on the top-left coordinates and scales
            if num_pixels > 1 and scale_x > 0 and scale_y > 0:
                # Calculate offsets based on the separate scales for each dimension
                new_offset_x = selection_x % scale_x
                new_offset_y = selection_y % scale_y
                
                # Using separate scales for X and Y dimensions
                # Calculated offsets based on separate scales
                
                # Update the offset inputs with the new values
                self.offset_x_input.setValue(int(new_offset_x))
                self.offset_y_input.setValue(int(new_offset_y))
            
            # Update the pixel info label with the calculated values
            if hasattr(self, 'pixel_info_label'):
                offset_x = self.offset_x_input.value()
                offset_y = self.offset_y_input.value()
                
                if num_pixels > 1:
                    # Include information about different scales for each axis if they differ
                    if abs(scale_x - scale_y) > 0.01:
                        self.pixel_info_label.setText(
                            f"Selected area: {pixel_width}x{pixel_height} pixels\n"
                            f"Contains {num_pixels}x{num_pixels} pixels\n"
                            f"Scale: X:{scale_x:.2f}, Y:{scale_y:.2f}\n"
                            f"Grid offset: ({offset_x}, {offset_y}) pixels"
                        )
                    else:
                        self.pixel_info_label.setText(
                            f"Selected area: {pixel_width}x{pixel_height} pixels\n"
                            f"Contains {num_pixels}x{num_pixels} pixels (scale: {scale_x:.2f})\n"
                            f"Grid offset: ({offset_x}, {offset_y}) pixels"
                        )
                else:
                    self.pixel_info_label.setText(
                        f"Selected 1 pixel of size {pixel_width}x{pixel_height}\n"
                        f"Grid offset: ({offset_x}, {offset_y}) pixels"
                    )
    
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
        # Only update if we have an image and selection size is set
        if not self.current_image_path or self.selection_size_input.value() <= 0:
            return
        
        # Show loading state
        self.preview_viewer.setText("Processing preview...")
        QApplication.processEvents()
            
        try:
            # Get the current settings
            selection_size = self.selection_size_input.value()
            num_pixels = self.num_pixels_input.value()
            
            # Calculate the scale (selection size / number of pixels)
            # If num_pixels is 0, treat selection as 1 pixel (1:1 mapping)
            scale_to_use = selection_size / max(1, num_pixels)
            
            # Get separate scales for X and Y if they exist
            if hasattr(self, 'scale_x') and hasattr(self, 'scale_y'):
                scale_x = self.scale_x
                scale_y = self.scale_y
            else:
                # If we don't have separate scales, use the same scale for both
                scale_x = scale_to_use
                scale_y = scale_to_use
            
            offset_x = self.offset_x_input.value()
            offset_y = self.offset_y_input.value()
            color_threshold = self.color_threshold_slider.value()
            use_median = self.use_median_check.isChecked()
            
            # Determine ignore_outer_pixels value based on checkbox and slider
            if self.ignore_outer_check.isChecked():
                ignore_outer_pixels = self.ignore_outer_slider.value()  # Get percentage (0-90)
                if ignore_outer_pixels == 0:  # If slider is at 0%, treat as True for backward compatibility
                    ignore_outer_pixels = True
            else:
                ignore_outer_pixels = False
            
            # Load the image
            img = Image.open(self.current_image_path)
            width, height = img.size
            
            # Apply offset if needed
            if offset_x > 0 or offset_y > 0:
                img = img.crop((offset_x, offset_y, width, height))
            
            # Format ignore_outer_pixels value for debug display
            if isinstance(ignore_outer_pixels, (int, float)) and ignore_outer_pixels != True:
                ignore_display = f"{ignore_outer_pixels}%"
            else:
                ignore_display = str(ignore_outer_pixels)
                
            
            # Safety check for small images
            if int(width / scale_x) < 1 or int(height / scale_y) < 1:
                self.preview_viewer.setText("Image too small for selected scale")
                return
            
            # Create a downscaled preview using the current settings
            try:
                # Add a safety check - if advanced color processing is selected, ensure clustering works
                if not use_median:
                    # If color clustering is likely to fail (with very small scale), use simple color processing
                    if min(scale_x, scale_y) <= 2:
                        # For very small scales, force simple color processing as a safer alternative
                        use_median = True
                        self.status_bar.showMessage("Using simple color processing (safer for very small scales)")
                        QApplication.processEvents()
                
                # For preview, we're applying the offset directly to the image object above
                # rather than creating a temporary file, so we pass the image that already has the offset applied
                
                # Get aspect ratio if we have one stored
                aspect_ratio = getattr(self, 'pixel_aspect_ratio', 1.0)
                
                # Get preserve aspect ratio setting
                preserve_aspect_ratio = self.preserve_aspect_check.isChecked()

                # If we're using simple color processing, pass color_threshold of 0 to disable clustering
                # Otherwise, use the selected threshold for advanced color processing
                effective_threshold = 0 if use_median else color_threshold
                
                preview_img = self.downscaler.downscale_image(
                    img,  # This has offset and aspect ratio correction applied if needed
                    scale_x, 
                    scale_y,
                    color_threshold=effective_threshold, 
                    use_median=use_median,
                    ignore_outer_pixels=ignore_outer_pixels
                )
                
                # Update the status
                self.status_bar.showMessage(f"Preview created at {preview_img.width}x{preview_img.height} pixels " +
                                           f"(scale: {scale_to_use:.2f})")
                
                # Display the preview in the viewer
                self.preview_viewer.setImage(preview_img)
                
                # Scale the preview zoom to match visual size
                orig_zoom = self.drop_area.zoom_factor
                calculated_scale = scale_to_use  # Using the scale we calculated above
                
                # For the preview, we want to apply the same relative zoom
                # If scale_to_use is N, each pixel in the preview represents N pixels in the original
                # So we need N times more zoom to maintain the same visual size
                preview_zoom = orig_zoom * calculated_scale
                
                # Ensure the zoom is at least 0.1
                preview_zoom = max(0.1, preview_zoom)
                self.preview_viewer.zoom_factor = preview_zoom
                
                # Update the UI to reflect the current zoom level of the input image
                self.zoom_label.setText(f"Zoom: {orig_zoom:.2f}x")
                
                # Apply the pan offsets to maintain the same visual position
                self.preview_viewer.pan_offset_x = self.drop_area.pan_offset_x
                self.preview_viewer.pan_offset_y = self.drop_area.pan_offset_y
                
                # Update both views
                self.drop_area.update()
                self.preview_viewer.update()
                
            except Exception as inner_e:
                self.preview_viewer.setText(f"Error creating preview: {str(inner_e)}")
                return
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview update error: {str(e)}")
            self.preview_viewer.setText(f"Error updating preview: {str(e)}")
    
    def onPixelSelected(self, pixel_width, pixel_height, offset_x, offset_y):
        """
        Handle pixel selection from the drag area.
        
        Parameters:
        - pixel_width: Width of the selected pixel(s) in pixels
        - pixel_height: Height of the selected pixel(s) in pixels
        - offset_x: X offset of the grid from the image edge
        - offset_y: Y offset of the grid from the image edge
        """
        # Store the raw dimensions - we'll work with these directly
        # Instead of normalizing them, we'll keep the separate scales for x and y
        
        # Calculate the aspect ratio for informational purposes
        aspect_ratio = pixel_width / pixel_height if pixel_height > 0 else 1.0
        
        # Store scales directly instead of trying to normalize dimensions
        scale_x = pixel_width  # For single pixel selection (num_pixels=1)
        scale_y = pixel_height
        
        # Store for future use
        self.scale_x = scale_x
        self.scale_y = scale_y
        
        # Grid visualization removed
        
        # Update UI values
        self.selection_size_input.setValue(max(pixel_width, pixel_height))  # Use the larger dimension for input
        self.offset_x_input.setValue(offset_x)
        self.offset_y_input.setValue(offset_y)
        
        # Store aspect ratio for informational purposes
        self.pixel_aspect_ratio = aspect_ratio
        
        # Create a label to display the aspect ratio in the UI
        if not hasattr(self, 'aspect_ratio_label'):
            # Add a new label to the UI next to the scale display
            from PySide6.QtWidgets import QLabel
            self.aspect_ratio_label = QLabel(f"Pixel Aspect Ratio: {aspect_ratio:.2f}")
            # Find the parent layout of the scale_display if possible
            for widget in self.findChildren(QLabel):
                if widget.text().startswith("Calculated Scale"):
                    # Get the parent layout of this widget
                    if widget.parent() and hasattr(widget.parent(), 'layout'):
                        parent_layout = widget.parent().layout()
                        if parent_layout:
                            # Add the new label to the layout
                            parent_layout.addWidget(self.aspect_ratio_label)
                            break
        else:
            # Update the existing label
            self.aspect_ratio_label.setText(f"Pixel Aspect Ratio: {aspect_ratio:.2f}")
        
        # Calculate the actual scale to use based on the selection size and number of pixels
        num_pixels = self.num_pixels_input.value()
        if num_pixels > 0:
            calculated_scale = pixel_width / num_pixels
            # Update the scale display (read-only, calculated)
            self.scale_display.setText(f"{calculated_scale:.2f}")
            
            # Get the top-left coordinates of the selection rectangle
            selection_x = 0
            selection_y = 0
            if hasattr(self.drop_area, 'selection_rect') and not self.drop_area.selection_rect.isEmpty():
                selection_x = self.drop_area.selection_rect.x()
                selection_y = self.drop_area.selection_rect.y()
            
            # If using multiple pixels (fractional scaling), recalculate offsets properly
            if num_pixels > 1:
                # Calculate the proper offsets based on the selection's top-left coordinates
                # and the calculated scale (effective pixel size)
                # Adjust for aspect ratio if needed
                if abs(aspect_ratio - 1.0) > 0.01:  # Non-square pixels
                    if aspect_ratio > 1.0:  # Wider than tall
                        # Horizontal dimension is stretched, adjust x calculation
                        adjusted_scale_x = calculated_scale / aspect_ratio
                        adjusted_scale_y = calculated_scale
                        new_offset_x = selection_x % adjusted_scale_x
                        new_offset_y = selection_y % adjusted_scale_y
                    else:  # Taller than wide
                        # Vertical dimension is stretched, adjust y calculation
                        adjusted_scale_x = calculated_scale
                        adjusted_scale_y = calculated_scale * aspect_ratio
                        new_offset_x = selection_x % adjusted_scale_x
                        new_offset_y = selection_y % adjusted_scale_y
                    
                else:
                    # Square pixels, standard calculation
                    new_offset_x = selection_x % calculated_scale
                    new_offset_y = selection_y % calculated_scale
                
                # Update the offset values
                self.offset_x_input.setValue(int(new_offset_x))
                self.offset_y_input.setValue(int(new_offset_y))
                
                # Update info label with all the details
                if abs(aspect_ratio - 1.0) > 0.01:  # If non-square (with a small tolerance)
                    self.pixel_info_label.setText(
                        f"Selected area: {pixel_width}×{pixel_height} pixels (non-square, ratio: {aspect_ratio:.2f})\n"
                        f"Contains {num_pixels} pixels (scale: {calculated_scale:.2f})\n"
                        f"Grid offset: ({int(new_offset_x)}, {int(new_offset_y)}) pixels"
                    )
                else:
                    self.pixel_info_label.setText(
                        f"Selected area: {pixel_width}×{pixel_height} pixels\n"
                        f"Contains {num_pixels} pixels (scale: {calculated_scale:.2f})\n"
                        f"Grid offset: ({int(new_offset_x)}, {int(new_offset_y)}) pixels"
                    )
            else:
                # For single pixel, use original offsets
                if abs(aspect_ratio - 1.0) > 0.01:  # If non-square (with a small tolerance)
                    self.pixel_info_label.setText(
                        f"Selected pixel size: {pixel_width}×{pixel_height} pixels (non-square, ratio: {aspect_ratio:.2f})\n"
                        f"Grid offset: ({offset_x}, {offset_y}) pixels"
                    )
                else:
                    self.pixel_info_label.setText(
                        f"Selected pixel size: {pixel_width}×{pixel_height} pixels\n"
                        f"Grid offset: ({offset_x}, {offset_y}) pixels"
                    )
        else:
            # If num_pixels is 0, default to treating selection as 1 pixel
            self.scale_display.setText(f"{pixel_width:.2f}")
            
            # Update info label with original offsets
            if abs(aspect_ratio - 1.0) > 0.01:  # If non-square
                self.pixel_info_label.setText(
                    f"Selected pixel size: {pixel_width}×{pixel_height} pixels (non-square, ratio: {aspect_ratio:.2f})\n"
                    f"Grid offset: ({offset_x}, {offset_y}) pixels"
                )
            else:
                self.pixel_info_label.setText(
                    f"Selected pixel size: {pixel_width}×{pixel_height} pixels\n"
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
            
            # Calculate the scale based on selection size and number of pixels
            selection_size = self.selection_size_input.value() if hasattr(self, 'selection_size_input') else 1
            num_pixels = self.num_pixels_input.value() if hasattr(self, 'num_pixels_input') else 1
            scale_to_use = selection_size / max(1, num_pixels)
            
            # Set initial zoom level for both views
            orig_zoom = 2  # Set input image to 2x by default
            self.drop_area.zoom_factor = orig_zoom
            self.drop_area.pan_offset_x = 0
            self.drop_area.pan_offset_y = 0
            
            # Scale preview zoom to match visual size
            # Use scale_to_use as the preview scale factor
            self.preview_viewer.zoom_factor = orig_zoom * scale_to_use
            self.preview_viewer.pan_offset_x = 0
            self.preview_viewer.pan_offset_y = 0
            
            # Update zoom label
            self.zoom_label.setText(f"Zoom: {self.drop_area.zoom_factor:.2f}x")
            
            # Connect pixel selection signal if not already connected
            try:
                self.drop_area.pixelSelected.disconnect()
            except:
                pass
            self.drop_area.pixelSelected.connect(self.onPixelSelected)
            
            # Enable the process button
            self.process_button.setEnabled(True)
            self.status_bar.showMessage(f"Loaded: {os.path.basename(file_path)}. Select a pixel or adjust settings manually.")
            
            # Set up the preview area with an initial blank image (don't show test image if loading a real image)
            blank_img = Image.new("RGBA", (100, 100), (240, 240, 240, 255))
            self.preview_viewer.setImage(blank_img)
            self.preview_viewer.setText("Select a pixel above and click 'Update Preview'")
            
        except Exception as e:
            self.status_bar.showMessage("Error loading image")
            QMessageBox.critical(self, "Error", f"An error occurred: {str(e)}")
    
    def toggleGridVisibility(self, visible):
        """Toggle the visibility of the grid overlay in the selection rectangle"""
        if hasattr(self, 'drop_area'):
            self.drop_area.show_grid = visible
            self.drop_area.update()  # Force redraw
    
    def showTestPreview(self):
        """Show a test image in the preview to verify display works"""
        try:
            # Default to a scale of 2 for a clear test image
            scale_x = scale_y = 2.0
            scale_to_use = 2.0  # Keep for backward compatibility
            
            # Create a small original test image (for the input view)
            orig_size = 4
            test_orig = Image.new("RGBA", (orig_size, orig_size), (255, 255, 255, 255))
            
            # Add a clear checkerboard pattern
            test_orig.putpixel((0, 0), (255, 0, 0, 255))  # Red
            test_orig.putpixel((1, 1), (255, 0, 0, 255))  # Red
            test_orig.putpixel((2, 2), (255, 0, 0, 255))  # Red
            test_orig.putpixel((3, 3), (255, 0, 0, 255))  # Red
            test_orig.putpixel((0, 2), (0, 0, 255, 255))  # Blue
            test_orig.putpixel((1, 3), (0, 0, 255, 255))  # Blue
            test_orig.putpixel((2, 0), (0, 0, 255, 255))  # Blue
            test_orig.putpixel((3, 1), (0, 0, 255, 255))  # Blue
            
            # Scale up for better visibility (this doesn't affect the scaling logic, just makes it visible)
            display_scale = 20
            test_orig = test_orig.resize((orig_size * display_scale, orig_size * display_scale), Image.NEAREST)
            
            # Load the original test image in the original viewer
            self.drop_area.image = test_orig
            self.drop_area.updatePixmap()
            
            # Create a scaled-down version (preview) - half the size to simulate downscaling
            scaled_size = int(orig_size / scale_to_use)
            test_scaled = Image.new("RGBA", (scaled_size, scaled_size), (255, 255, 255, 255))
            
            # Add a matching pattern, appropriately scaled down
            test_scaled.putpixel((0, 0), (255, 0, 0, 255))  # Red
            test_scaled.putpixel((1, 1), (0, 0, 255, 255))  # Blue
            
            # Scale up for better visibility
            test_scaled = test_scaled.resize((scaled_size * display_scale, scaled_size * display_scale), Image.NEAREST)
            
            # Update preview viewer
            self.preview_viewer.setImage(test_scaled)
            
            # Set zoom levels to maintain visual size ratio
            orig_zoom = 2.0  # Default input zoom
            self.drop_area.zoom_factor = orig_zoom
            self.preview_viewer.zoom_factor = orig_zoom * scale_to_use
            
            # Reset pan offsets
            self.drop_area.pan_offset_x = 0
            self.drop_area.pan_offset_y = 0
            self.preview_viewer.pan_offset_x = 0
            self.preview_viewer.pan_offset_y = 0
            
            # Update zoom label
            self.zoom_label.setText(f"Zoom: {orig_zoom:.2f}x")
            
            # Set the scales in the class instance for other methods to use
            self.scale_x = scale_x
            self.scale_y = scale_y
            
            # Update scale value in UI to match the test scale
            if hasattr(self, 'selection_size_input') and hasattr(self, 'num_pixels_input'):
                self.selection_size_input.setValue(int(scale_to_use))
                self.num_pixels_input.setValue(1)
            
            self.status_bar.showMessage(f"Test pattern displayed. Scale: {scale_to_use}x (each preview pixel = {scale_to_use} input pixels)")
            
        except Exception as e:
            # Test preview error occurred
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
            
            # Get the selection size, number of pixels, and calculate scales
            num_pixels = self.num_pixels_input.value()
            selection_size = self.selection_size_input.value()
            
            # Get separate scales for X and Y if available
            if hasattr(self, 'scale_x') and hasattr(self, 'scale_y'):
                scale_x = self.scale_x
                scale_y = self.scale_y
            else:
                # Fall back to using selection_size if separate scales aren't set
                scale_x = scale_y = selection_size / max(1, num_pixels)
            
            # For status updates
            if abs(scale_x - scale_y) > 0.01:
                scale_display = f"X:{scale_x:.2f}, Y:{scale_y:.2f}"
            else:
                scale_display = f"{scale_x:.2f}"
            
            offset_x = self.offset_x_input.value()
            offset_y = self.offset_y_input.value()
            
            # Get other options
            export_original_size = self.orig_size_check.isChecked()
            color_threshold = self.color_threshold_slider.value()
            use_median = self.use_median_check.isChecked()
            
            # Determine ignore_outer_pixels value based on checkbox and slider
            if self.ignore_outer_check.isChecked():
                ignore_outer_pixels = self.ignore_outer_slider.value()  # Get percentage (0-90)
                if ignore_outer_pixels == 0:  # If slider is at 0%, treat as True for backward compatibility
                    ignore_outer_pixels = True
            else:
                ignore_outer_pixels = False
                
            custom_upscale_factor = None
            if self.custom_upscale_check.isChecked():
                custom_upscale_factor = self.custom_upscale.value()
            
            # Show scale info in status bar
            if num_pixels > 1:
                self.status_bar.showMessage(
                    f"Using scale: {scale_display} (selection size {selection_size} / {num_pixels} pixels)"
                    f" with offset ({offset_x}, {offset_y})"
                )
            else:
                self.status_bar.showMessage(
                    f"Using pixel size: {scale_display} with offset ({offset_x}, {offset_y})"
                )
            QApplication.processEvents()
            
            
            # Safety check - if not using median, make sure it's safe
            if not use_median:
                # If color clustering is likely to fail (with very small scale), use simple mean instead
                if min(scale_x, scale_y) <= 2:
                    # For very small scales, force use_median to True as a safer alternative
                    use_median = True
                    # Inform the user that we're using median instead
                    self.status_bar.showMessage("Using median color for small scale (safer for small scales)")
                    QApplication.processEvents()
            
            # When using advanced color processing (not median), add color threshold details to status message
            if not use_median:
                # Safety check - if color clustering might fail with very small scale
                if min(scale_x, scale_y) <= 2:
                    # For very small scales, force use_median to True as a safer alternative
                    use_median = True
                    # Inform the user that we're using simple processing instead
                    self.status_bar.showMessage("Using simple color processing for small scale (safer for small scales)")
                    QApplication.processEvents()
                else:
                    # Add details about the color threshold being used
                    threshold_info = f" with color similarity threshold: {color_threshold}"
                    self.status_bar.showMessage(self.status_bar.currentMessage() + threshold_info)
                    QApplication.processEvents()
            
            # If we're using simple color processing, pass color_threshold of 0 to disable clustering
            # Otherwise, use the selected threshold for advanced color processing
            effective_threshold = 0 if use_median else color_threshold
            
            # Get aspect ratio if we have one stored
            aspect_ratio = getattr(self, 'pixel_aspect_ratio', 1.0)
            
            # Get preserve aspect ratio setting
            preserve_aspect_ratio = self.preserve_aspect_check.isChecked()
            
            # First crop the image if there's an offset
            if offset_x > 0 or offset_y > 0:
                # Crop to align with the grid
                pil_img = pil_img.crop((offset_x, offset_y, width, height))
                # Image cropped to align with pixel grid
                # Create a temporary path for the cropped image
                temp_dir = os.path.dirname(file_path)
                temp_file = os.path.join(temp_dir, f"temp_cropped_{os.path.basename(file_path)}")
                pil_img.save(temp_file)
                file_path = temp_file
            
            # If we're using a temporary file for the cropped image, we've already applied the offset
            # Otherwise, we need to pass the offset values to the process_image method
            if offset_x > 0 or offset_y > 0:
                downscaled_path, clean_path = self.downscaler.process_image(
                    file_path,  # This will be the temp_file with offset already applied
                    force_scale_x=scale_x,
                    force_scale_y=scale_y,
                    upscale_factor=None,  # Will be auto-calculated in the function
                    export_original_size=export_original_size,
                    color_threshold=effective_threshold,
                    use_median=use_median,
                    ignore_outer_pixels=ignore_outer_pixels,
                    preserve_original_proportions=preserve_aspect_ratio  # Whether to keep original proportions during upscaling
                )
            else:
                # No offset needed
                downscaled_path, clean_path = self.downscaler.process_image(
                    file_path,
                    force_scale_x=scale_x,
                    force_scale_y=scale_y,
                    upscale_factor=None,  # Will be auto-calculated in the function
                    export_original_size=export_original_size,
                    color_threshold=effective_threshold,
                    use_median=use_median,
                    ignore_outer_pixels=ignore_outer_pixels,
                    preserve_original_proportions=preserve_aspect_ratio  # Whether to keep original proportions during upscaling
                )
            
            # Process additional custom upscale if requested
            if custom_upscale_factor and custom_upscale_factor > 1:
                self.status_bar.showMessage(f"Creating {custom_upscale_factor}x upscaled version...")
                QApplication.processEvents()
                
                # Important: Use the same file_path as the main processing to ensure
                # consistent offset handling across both operations
                custom_downscaled, custom_clean = self.downscaler.process_image(
                    file_path,  # This will be the temp_file with offset applied if offset was present
                    force_scale_x=scale_x,  # Use the same scales as main processing
                    force_scale_y=scale_y,
                    upscale_factor=custom_upscale_factor,
                    export_original_size=False,
                    color_threshold=effective_threshold,  # Use the same effective threshold as main processing
                    use_median=use_median,
                    ignore_outer_pixels=ignore_outer_pixels,
                    preserve_original_proportions=preserve_aspect_ratio  # Whether to keep original proportions during upscaling
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
                    
                        # Calculate the scale based on selection size and number of pixels
                    selection_size = self.selection_size_input.value()
                    num_pixels = self.num_pixels_input.value()
                    scale_to_use = selection_size / max(1, num_pixels)
                    
                    # Scale the zoom to match visual size
                    orig_zoom = self.drop_area.zoom_factor
                    self.preview_viewer.zoom_factor = min(10, max(1, int(orig_zoom * scale_to_use)))
                    self.preview_viewer.update()
                    
                    # Update the status
                    self.status_bar.showMessage(f"Final clean version displayed in preview")
                except Exception as view_err:
                    # Failed to display clean result
                    pass
            
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