import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np

class PixelArtDownscaler:
    def __init__(self, root):
        self.root = root
        self.root.title("Pixel Art Downscaler")
        self.root.geometry("500x400")
        
        # Set up the drop area
        self.drop_area = tk.Frame(root, bg="#f0f0f0", width=400, height=300)
        self.drop_area.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.label = tk.Label(self.drop_area, text="Drag and drop your pixel art image here", bg="#f0f0f0")
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Enable drag and drop
        self.drop_area.drop_target_register(tk.DND_FILES)
        self.drop_area.dnd_bind('<<Drop>>', self.process_drop)
        
        # Status message
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_label = tk.Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def process_drop(self, event):
        file_path = event.data
        
        # Clean up the file path (handle platform specific issues)
        if file_path.startswith('{'):
            file_path = file_path[1:]
        if file_path.endswith('}'):
            file_path = file_path[:-1]
            
        # On Windows, the path might include curly braces and multiple file paths
        if os.name == 'nt':
            file_path = file_path.replace('{', '').replace('}', '')
            # If multiple files were dropped, just use the first one
            if ' ' in file_path:
                file_path = file_path.split(' ')[0]
        
        self.status_var.set(f"Processing: {os.path.basename(file_path)}")
        self.root.update()
        
        self.process_image(file_path)
    
    def process_image(self, file_path):
        try:
            # Load the image
            img = Image.open(file_path)
            img = img.convert("RGBA")  # Ensure alpha channel is preserved
            
            # Detect the pixel scale
            pixel_scale = self.detect_pixel_scale(img)
            if pixel_scale <= 1:
                messagebox.showinfo("No scaling needed", "This image is already scaled to 1:1 pixel ratio.")
                self.status_var.set("Ready")
                return
                
            # Downscale the image
            downscaled_img = self.downscale_image(img, pixel_scale)
            
            # Save the downscaled image
            output_path = self.get_output_path(file_path)
            downscaled_img.save(output_path)
            
            messagebox.showinfo("Success", f"Image downscaled (scale factor: {pixel_scale})\nSaved as: {os.path.basename(output_path)}")
            self.status_var.set(f"Saved: {os.path.basename(output_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
            self.status_var.set("Error occurred")
    
    def detect_pixel_scale(self, img):
        """
        Detect the pixel scale by analyzing the image for repeated patterns.
        Returns the detected scale (an integer representing how many pixels make up one 'art pixel').
        """
        # Simple implementation: check for vertical and horizontal color patterns
        width, height = img.size
        
        # Convert image to numpy array for easier analysis
        img_array = np.array(img)
        
        # Start with a minimum scale of 2 (1 would mean no scaling needed)
        min_scale = 2
        max_scale = min(width // 4, height // 4, 20)  # Limit the max scale to check
        
        best_scale = 1  # Default if no pattern is found
        
        # Check horizontal patterns
        for scale in range(min_scale, max_scale + 1):
            horizontal_match = True
            vertical_match = True
            
            # Sample several rows and columns to check for patterns
            sample_points = min(20, min(width, height) // scale)
            
            # Check horizontal pattern (rows)
            for y in range(0, min(height, scale * sample_points), scale):
                for x in range(0, width - scale, scale):
                    for dx in range(scale):
                        if not np.array_equal(img_array[y, x], img_array[y, x + dx]):
                            horizontal_match = False
                            break
                    if not horizontal_match:
                        break
                if not horizontal_match:
                    break
            
            # Check vertical pattern (columns)
            for x in range(0, min(width, scale * sample_points), scale):
                for y in range(0, height - scale, scale):
                    for dy in range(scale):
                        if not np.array_equal(img_array[y, x], img_array[y + dy, x]):
                            vertical_match = False
                            break
                    if not vertical_match:
                        break
                if not vertical_match:
                    break
            
            if horizontal_match or vertical_match:
                best_scale = scale
                break
        
        return best_scale
    
    def downscale_image(self, img, scale):
        """
        Downscale the image by the given scale factor using average color.
        """
        width, height = img.size
        new_width = width // scale
        new_height = height // scale
        
        # Create a new image for the result
        result = Image.new("RGBA", (new_width, new_height))
        
        # Convert to numpy array for efficient processing
        img_array = np.array(img)
        
        for y in range(new_height):
            for x in range(new_width):
                # Extract the block of pixels that make up this 'art pixel'
                block = img_array[y*scale:(y+1)*scale, x*scale:(x+1)*scale]
                
                # Calculate the average color (per channel)
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                
                # Set the pixel in the result image
                result.putpixel((x, y), tuple(avg_color))
        
        return result
    
    def get_output_path(self, input_path):
        """Generate an output path for the downscaled image."""
        dir_name = os.path.dirname(input_path)
        file_name = os.path.basename(input_path)
        name, ext = os.path.splitext(file_name)
        
        output_name = f"{name}_downscaled.png"
        output_path = os.path.join(dir_name, output_name)
        
        # Ensure we don't overwrite existing files
        counter = 1
        while os.path.exists(output_path):
            output_name = f"{name}_downscaled_{counter}.png"
            output_path = os.path.join(dir_name, output_name)
            counter += 1
            
        return output_path

# Add TkinterDnD2 import and setup
def main():
    try:
        import TkinterDnD2
        root = TkinterDnD2.Tk()
    except ImportError:
        messagebox.showerror("Missing Dependency", "This app requires TkinterDnD2. Please install it using:\npip install tkinterdnd2")
        exit(1)
    
    app = PixelArtDownscaler(root)
    root.mainloop()

if __name__ == "__main__":
    main()