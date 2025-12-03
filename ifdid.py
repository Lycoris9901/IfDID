#!/usr/bin/env python3
"""
IFDID - Intelligent Face Detection for Identifying Differences
A GUI application to analyze faces for potential cosmetic alterations.
"""

import importlib
import subprocess
import sys
import io
import os
import json
import threading
import time
import gc
import re

REQUIRED_MODULES = {
    "customtkinter": "customtkinter",
    "PIL": "Pillow",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "google": "google-genai",
    "tkinterdnd2": "tkinterdnd2"
}


def ensure_module(module_name, package_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        print(f"[IFDID] Installing missing dependency: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])


for module, package in REQUIRED_MODULES.items():
    ensure_module(module, package)

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance, ImageOps
import cv2
import numpy as np
from google import genai
from google.genai import types
from tkinterdnd2 import TkinterDnD, DND_FILES

# Configure appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
ctk.set_window_scaling(0.9)
ctk.set_widget_scaling(1.2)


class Tk(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TkdndVersion = TkinterDnD._require(self)


class IFDIDApp(Tk):
    def __init__(self):
        super().__init__()
        
        self.title("IFDID - Face Analysis")
        self.geometry("1200x900")
        self.minsize(1100, 820)
        self.resizable(False, False)
        
        # Configure colors
        self.colors = {
            "bg_dark": "#0a0a0f",
            "bg_card": "#14141f",
            "bg_hover": "#1e1e2e",
            "accent": "#6366f1",
            "accent_hover": "#818cf8",
            "text": "#e2e8f0",
            "text_dim": "#64748b",
            "green": "#22c55e",
            "orange": "#f97316",
            "red": "#ef4444",
            "border": "#2d2d3d"
        }
        
        self.configure(fg_color=self.colors["bg_dark"])
        
        self.image_path = None
        self.original_image = None
        self.analyzed_image = None
        self.analysis_result = None
        self.processing_image = None
        
        self.setup_ui()
        self.setup_drag_drop()
        
    def setup_ui(self):
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=24, pady=24)
        self.main_frame.grid_columnconfigure((0, 1), weight=1)
        self.main_frame.grid_rowconfigure(0, weight=1)
        
        # Left column
        self.left_panel = ctk.CTkFrame(
            self.main_frame,
            fg_color=self.colors["bg_card"],
            corner_radius=18,
            border_width=1,
            border_color=self.colors["border"],
            width=380
        )
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.left_panel.grid_rowconfigure(2, weight=1)
        
        self.brand_label = ctk.CTkLabel(
            self.left_panel,
            text="IFDID",
            font=ctk.CTkFont(family="SF Pro Display", size=38, weight="bold"),
            text_color=self.colors["bg_dark"],
            fg_color=self.colors["accent"],
            corner_radius=14,
            padx=14,
            pady=4
        )
        self.brand_label.pack(anchor="w", padx=20, pady=(16, 4))
        
        self.tagline_label = ctk.CTkLabel(
            self.left_panel,
            text="Native beauty intelligence",
            font=ctk.CTkFont(size=16),
            text_color=self.colors["text_dim"]
        )
        self.tagline_label.pack(anchor="w", padx=20, pady=(0, 6))
        
        self.input_header = ctk.CTkLabel(
            self.left_panel,
            text="Upload Photo",
            font=ctk.CTkFont(size=18),
            text_color=self.colors["text_dim"]
        )
        self.input_header.pack(anchor="w", padx=20, pady=(0, 8))
        
        self.drop_frame = ctk.CTkFrame(
            self.left_panel,
            fg_color=self.colors["bg_dark"],
            corner_radius=14,
            border_width=2,
            border_color=self.colors["border"],
            height=360
        )
        self.drop_frame.pack(fill="both", expand=True, padx=20, pady=(0, 12))
        self.drop_frame.pack_propagate(False)
        
        self.drop_label = ctk.CTkLabel(
            self.drop_frame,
            text="🖼️ Drop a face photo here\n(Photos drag supported)",
            font=ctk.CTkFont(size=18),
            text_color=self.colors["text_dim"],
            justify="center",
            wraplength=360
        )
        self.drop_label.pack(expand=True, padx=12, pady=12)
        
        self.image_label = ctk.CTkLabel(self.drop_frame, text="")
        self.image_label.pack(expand=True)
        self.image_label.pack_forget()
        
        self.settings_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.settings_frame.pack(fill="x", padx=20, pady=(0, 12))
        
        self.settings_frame.columnconfigure((0, 1), weight=1)
        
        self.thinking_label = ctk.CTkLabel(
            self.settings_frame,
            text="Reasoning",
            font=ctk.CTkFont(size=15),
            text_color=self.colors["text_dim"]
        )
        self.thinking_label.grid(row=0, column=0, sticky="w")
        
        self.thinking_var = ctk.StringVar(value="Low")
        self.thinking_menu = ctk.CTkOptionMenu(
            self.settings_frame,
            variable=self.thinking_var,
            values=["Low", "Medium", "High"],
            width=140,
            fg_color=self.colors["bg_dark"],
            button_color=self.colors["accent"],
            button_hover_color=self.colors["accent_hover"]
        )
        self.thinking_menu.grid(row=0, column=1, sticky="e")
        
        self.start_button = ctk.CTkButton(
            self.left_panel,
            text="⚡ Start Analysis",
            font=ctk.CTkFont(size=18, weight="bold"),
            height=104,
            corner_radius=12,
            fg_color=self.colors["accent"],
            hover_color=self.colors["accent_hover"],
            command=self.start_analysis
        )
        self.start_button.pack(fill="x", padx=20, pady=(0, 18))
        
        # Right column
        self.right_panel = ctk.CTkFrame(
            self.main_frame,
            fg_color=self.colors["bg_card"],
            corner_radius=18,
            border_width=1,
            border_color=self.colors["border"]
        )
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        self.right_panel.grid_rowconfigure(3, weight=1)
        
        self.title_bar = ctk.CTkFrame(self.right_panel, fg_color="transparent", height=8)
        self.title_bar.pack(fill="x", padx=24, pady=(18, 0))
        
        self.score_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.score_frame.pack(fill="x", padx=24, pady=(10, 6))
        
        self.score_title = ctk.CTkLabel(
            self.score_frame,
            text="How likely they did cosmetic surgery?",
            font=ctk.CTkFont(size=18),
            text_color=self.colors["text_dim"]
        )
        self.score_title.pack()
        
        self.score_label = ctk.CTkLabel(
            self.score_frame,
            text="--",
            font=ctk.CTkFont(family="SF Mono", size=64, weight="bold"),
            text_color=self.colors["text_dim"]
        )
        self.score_label.pack()
        
        self.progress_frame = ctk.CTkFrame(self.score_frame, fg_color=self.colors["bg_dark"], height=10, corner_radius=5)
        self.progress_frame.pack(fill="x", pady=(8, 2))
        self.progress_frame.pack_propagate(False)
        
        self.progress_bar = ctk.CTkFrame(self.progress_frame, fg_color=self.colors["text_dim"], corner_radius=5)
        self.progress_bar.place(relx=0, rely=0, relheight=1, relwidth=0)
        
        self.result_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.colors["bg_dark"],
            corner_radius=16
        )
        self.result_frame.pack(fill="both", expand=True, padx=24, pady=16)
        self.result_frame.pack_propagate(False)
        
        self.result_image_label = ctk.CTkLabel(
            self.result_frame,
            text="Analysis will appear here",
            font=ctk.CTkFont(size=18),
            text_color=self.colors["text_dim"]
        )
        self.result_image_label.pack(expand=True)
        
        self.details_frame = ctk.CTkFrame(
            self.right_panel,
            fg_color=self.colors["bg_dark"],
            corner_radius=12,
            height=240
        )
        self.details_frame.pack(fill="x", padx=24, pady=(0, 20))
        self.details_frame.pack_propagate(False)
        
        self.details_textbox = ctk.CTkTextbox(
            self.details_frame,
            font=ctk.CTkFont(size=15),
            text_color=self.colors["text_dim"],
            fg_color="transparent",
            wrap="word"
        )
        self.details_textbox.pack(fill="both", expand=True, padx=12, pady=12)
        self.details_textbox.insert("0.0", "Detected alterations will be listed here after analysis.")
        self.details_textbox.configure(state="disabled")
        
        self.loading_frame = ctk.CTkFrame(self, fg_color=self.colors["bg_dark"])
        self.loading_label = ctk.CTkLabel(
            self.loading_frame,
            text="🔍 Analyzing...",
            font=ctk.CTkFont(size=28, weight="bold"),
            text_color=self.colors["accent"]
        )
        self.loading_label.pack(expand=True)
        
        self.current_tab = "analyzed"
        
    def setup_drag_drop(self):
        """Setup drag and drop functionality"""
        try:
            # Register the drop frame as a drop target
            self.drop_frame.drop_target_register(DND_FILES)
            self.drop_frame.dnd_bind('<<Drop>>', self.drop_image)
            
            # Also register the label inside just in case
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind('<<Drop>>', self.drop_image)
            
            self.image_label.drop_target_register(DND_FILES)
            self.image_label.dnd_bind('<<Drop>>', self.drop_image)
            
        except Exception as e:
            print(f"Drag and drop setup failed: {e}")
            
    def drop_image(self, event):
        """Handle dropped image"""
        filepath = event.data
        
        # Handle curly braces for paths with spaces (Windows/macOS Tkinter behavior)
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]
            
        # If multiple files dropped, take the first one
        if ' ' in filepath and not os.path.exists(filepath):
             # Try to split if it looks like multiple files (naive check)
             parts = filepath.split(' ')
             if os.path.exists(parts[0]):
                 filepath = parts[0]

        if os.path.isfile(filepath):
            # Check extension
            ext = os.path.splitext(filepath)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp']:
                self.load_image(filepath)
            else:
                messagebox.showwarning("Invalid File", "Please drop an image file.")
            
    def browse_image(self, event=None):
        """Open file dialog to browse for image"""
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.webp"),
            ("All files", "*.*")
        ]
        filepath = filedialog.askopenfilename(filetypes=filetypes)
        if filepath:
            self.load_image(filepath)
            
    def load_image(self, filepath):
        """Load and display the selected image"""
        try:
            self.image_path = filepath
            self.original_image = Image.open(filepath)
            
            # Resize for display
            display_size = (350, 350)
            img_display = self.original_image.copy()
            img_display.thumbnail(display_size, Image.Resampling.LANCZOS)
            
            photo = ctk.CTkImage(light_image=img_display, dark_image=img_display, size=img_display.size)
            
            self.drop_label.pack_forget()
            self.image_label.configure(image=photo)
            self.image_label.image = photo
            self.image_label.pack(expand=True)
            
            # Reset results
            self.reset_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
            
    def reset_results(self):
        """Reset the analysis results"""
        self.score_label.configure(text="--", text_color=self.colors["text_dim"])
        self.progress_bar.place(relwidth=0)
        self.result_image_label.configure(image=None, text="Analysis results will appear here")
        self.details_textbox.configure(state="normal")
        self.details_textbox.delete("0.0", "end")
        self.details_textbox.insert("0.0", "Detected alterations will be listed here after analysis.")
        self.details_textbox.configure(state="disabled")
        self.analyzed_image = None
        self.analysis_result = None
        self.processing_image = None
    
    def refresh_analysis_state(self):
        """Clear caches before a new analysis run"""
        self.analyzed_image = None
        self.analysis_result = None
        self.processing_image = None
        self.progress_bar.place(relwidth=0)
        self.details_textbox.configure(state="normal")
        self.details_textbox.delete("0.0", "end")
        self.details_textbox.insert("0.0", "Analyzing current photo...")
        self.details_textbox.configure(state="disabled")
        gc.collect()
        
    def get_score_color(self, score):
        """Get color based on score value"""
        if score < 60:
            # Green to Orange gradient
            ratio = score / 60
            r = int(34 + ratio * (249 - 34))
            g = int(197 + ratio * (115 - 197))
            b = int(94 + ratio * (22 - 94))
        elif score < 80:
            # Orange to Red gradient
            ratio = (score - 60) / 20
            r = int(249 + ratio * (239 - 249))
            g = int(115 + ratio * (68 - 115))
            b = int(22 + ratio * (68 - 22))
        else:
            # Red
            r, g, b = 239, 68, 68
            
        return f"#{r:02x}{g:02x}{b:02x}"
        
    def start_analysis(self):
        """Start the face analysis"""
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        
        # Refresh state for new run
        self.refresh_analysis_state()
        self.result_image_label.configure(image=None, text="Analyzing...")
            
        # Show loading
        self.loading_frame.place(relx=0.5, rely=0.5, anchor="center", relwidth=0.3, relheight=0.2)
        self.start_button.configure(state="disabled")
        
        # Run analysis in background thread
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()
        
    def run_analysis(self):
        """Run the actual analysis using Gemini API"""
        try:
            print("Starting analysis...")
            # Initialize Gemini client
            # Hardcoded API key as requested
            api_key = "AIzaSyB-2oAvdt6evGCjDhCdhbGgNrMXOuncHk4"
            
            # Use the new Client from google.genai
            client = genai.Client(api_key=api_key)
            self.api_key = api_key
            
            # Load image for Gemini
            print(f"Loading image: {self.image_path}")
            img = Image.open(self.image_path)
            
            # Face detection and cropping
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(img_cv, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
            
            self.face_offset = (0, 0)
            self.face_size = img.size
            processing_img = img
            
            if len(faces) > 0:
                # Find largest face
                faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                x, y, w, h = faces[0]
                
                # Add padding (20%)
                pad_w = int(w * 0.5)
                pad_h = int(h * 0.7)
                
                crop_x = max(0, x - pad_w)
                crop_y = max(0, y - pad_h)
                crop_w = min(img.width - crop_x, w + 2*pad_w)
                crop_h = min(img.height - crop_y, h + 2*pad_h)
                
                self.face_offset = (crop_x, crop_y)
                self.face_size = (crop_w, crop_h)
                
                processing_img = img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                print(f"Detected face, cropping to: {crop_w}x{crop_h}")
            else:
                print("No face detected via OpenCV, using full image")
            
            # Save processing image for UI display
            self.processing_image = processing_img.copy()

            # Resize image if too large to speed up processing
            max_size = 768
            if processing_img.width > max_size or processing_img.height > max_size:
                print(f"Resizing image from {processing_img.size} to max {max_size}px")
                processing_img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Analyze for plastic surgery indicators
            analysis_prompt = """Analyze the provided face for cosmetic/plastic surgery alterations (rhinoplasty, fillers, implants, bone shaving).
            
Return a JSON object with:
- score: 0-100 probability of alteration
- regions: list of detected alterations, each with "name", "description", "confidence", "x_percent" (0-100), "y_percent" (0-100), "radius_percent" (size)
- summary: text explanation
- native_description: guess of original features

Focus on structural anomalies. Ignore skin smoothing/makeup."""

            # Get thinking level from UI
            thinking_level = self.thinking_var.get().lower()
            print(f"Using thinking level: {thinking_level}")

            model_name = "gemini-3-pro-preview"
            print(f"Using model: {model_name}")

            # Compress image to reduce payload and build parts explicitly
            if processing_img.mode != "RGB":
                processing_img = processing_img.convert("RGB")

            with io.BytesIO() as buffer:
                processing_img.save(buffer, format="JPEG", quality=85, optimize=True)
                image_bytes = buffer.getvalue()
            # Use Gemini 3.0 Pro Preview with thinking config
            print("Calling Gemini 3.0 Pro Preview...")
            # Strictly use Gemini 3 Pro as requested - no fallback, but add retries
            max_retries = 2
            response = None
            for attempt in range(max_retries + 1):
                try:
                    user_content = [{
                        "role": "user",
                        "parts": [
                            {"text": analysis_prompt},
                            {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
                        ]
                    }]
                    response = client.models.generate_content(
                        model=model_name,
                        contents=user_content,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                            response_mime_type="application/json",
                            max_output_tokens=4096,
                            temperature=0.2
                        )
                    )
                    break
                except Exception as e:
                    err_text = str(e)
                    if attempt < max_retries and ("UNAVAILABLE" in err_text.upper() or "503" in err_text):
                        wait = 2 * (attempt + 1)
                        print(f"Gemini 3 Pro busy (attempt {attempt+1}/{max_retries+1}). Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            
            if response is None:
                raise RuntimeError("Gemini 3 Pro did not return a response.")
            
            print("Gemini response received")
            
            # Parse response
            response_text = getattr(response, "text", None)
            if not response_text:
                try:
                    response_text = response.candidates[0].content.parts[0].text
                except Exception:
                    response_text = ""

            def extract_json_block(text):
                if not text:
                    return None
                
                # Robust search for the outermost valid JSON object
                text = text.strip()
                
                # Try finding every possible '{' and matching '}'
                stack = []
                first_brace = -1
                
                for i, char in enumerate(text):
                    if char == '{':
                        if not stack:
                            first_brace = i
                        stack.append(char)
                    elif char == '}':
                        if stack:
                            stack.pop()
                            if not stack:
                                # Found a complete outer block
                                candidate = text[first_brace : i+1]
                                try:
                                    json.loads(candidate)
                                    return candidate
                                except json.JSONDecodeError:
                                    # If it fails, maybe we need to keep searching or it's malformed
                                    pass
                
                # Fallback: simple regex-like search if the stack method missed (e.g. unbalanced text)
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                if start_idx != -1 and end_idx > start_idx:
                    candidate = text[start_idx : end_idx+1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        pass
                
                return None

            json_payload = extract_json_block(response_text)
            
            if not json_payload:
                # Retry once with stronger instruction if no JSON found
                strict_prompt = analysis_prompt + "\n\nRemember: respond ONLY with the JSON object, nothing else."
                retry_content = [{
                    "role": "user",
                    "parts": [
                        {"text": strict_prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}}
                    ]
                }]
                retry_resp = client.models.generate_content(
                    model=model_name,
                    contents=retry_content,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                        response_mime_type="application/json",
                        max_output_tokens=4096,
                        temperature=0.2
                    )
                )
                retry_text = getattr(retry_resp, "text", None) or ""
                if not retry_text and retry_resp.candidates:
                    try:
                        retry_text = retry_resp.candidates[0].content.parts[0].text
                    except Exception:
                        retry_text = ""
                json_payload = extract_json_block(retry_text)
                if not json_payload:
                    response_text = response_text + "\n---\n" + retry_text
            
            if json_payload:
                try:
                    # Try standard parsing first
                    self.analysis_result = json.loads(json_payload)
                except json.JSONDecodeError:
                    # Try to fix common issues: trailing commas
                    try:
                        fixed_payload = re.sub(r',\s*([\]}])', r'\1', json_payload)
                        self.analysis_result = json.loads(fixed_payload)
                    except json.JSONDecodeError:
                        # Fallback: Regex extraction if JSON is hopeless
                        score_match = re.search(r'"score":\s*(\d+)', json_payload)
                        score_val = int(score_match.group(1)) if score_match else 50
                        
                        summary_match = re.search(r'"summary"\s*:\s*"(.*?)"\s*(?:,|\})', json_payload, re.DOTALL)
                        summary_val = summary_match.group(1) if summary_match else response_text
                        # Clean up newlines/quotes if regex grabbed too much
                        if summary_val and summary_val.endswith('"'):
                             summary_val = summary_val[:-1]
                        
                        self.analysis_result = {
                            "score": score_val,
                            "regions": [],
                            "summary": summary_val,
                            "native_description": "Unable to determine",
                            "raw": json_payload
                        }
            else:
                # No JSON block found, try regex on the whole text
                score_match = re.search(r'"score":\s*(\d+)', response_text)
                score_val = int(score_match.group(1)) if score_match else 50
                
                summary_match = re.search(r'"summary"\s*:\s*"(.*?)"\s*(?:,|\})', response_text, re.DOTALL)
                summary_val = summary_match.group(1) if summary_match else response_text
                
                self.analysis_result = {
                    "score": score_val,
                    "regions": [],
                    "summary": summary_val,
                    "native_description": "Unable to determine"
                }
            
            # Create analyzed image with circles
            self.create_analyzed_image()
            
            # Update UI
            self.after(0, self.update_results)
            
        except Exception as e:
            print(f"Analysis error: {e}")
            error_msg = str(e)
            self.after(0, lambda: self.show_error(error_msg))
            
    def create_analyzed_image(self):
        """Create image with circled regions"""
        if not getattr(self, 'processing_image', None) or not self.analysis_result:
            return
            
        # Use the cropped face image
        img = self.processing_image.copy()
        draw = ImageDraw.Draw(img)
        
        width, height = img.size
        
        # Draw circles on detected regions
        regions = self.analysis_result.get("regions", [])
        for region in regions:
            x_percent = region.get("x_percent", 50)
            y_percent = region.get("y_percent", 50)
            radius_percent = region.get("radius_percent", 5)
            
            # Calculate coordinates relative to the face crop (processing_image)
            x = int(width * x_percent / 100)
            y = int(height * y_percent / 100)
            radius = int(min(width, height) * radius_percent / 100)
            
            # Determine color based on confidence
            confidence = region.get("confidence", "medium")
            if confidence == "high":
                color = "#ef4444"  # Red
            elif confidence == "medium":
                color = "#f97316"  # Orange
            else:
                color = "#eab308"  # Yellow
                
            # Draw circle outline
            for offset in range(3):  # Thicker line
                draw.ellipse(
                    [x - radius - offset, y - radius - offset, 
                     x + radius + offset, y + radius + offset],
                    outline=color,
                    width=2
                )
                
        self.analyzed_image = img

    def update_results(self):
        """Update the UI with analysis results"""
        # Hide loading
        self.loading_frame.place_forget()
        self.start_button.configure(state="normal")
        
        if not self.analysis_result:
            return
            
        # Update score
        score = self.analysis_result.get("score", 0)
        color = self.get_score_color(score)
        
        score_display = int(round(float(score)))
        self.score_label.configure(text=str(score_display), text_color=color)
        self.progress_bar.configure(fg_color=color)
        self.progress_bar.place(relwidth=score/100)
        
        summary = self.analysis_result.get("summary", "")
        regions = self.analysis_result.get("regions", [])
        
        details_text = ""
        if summary:
            summary_text = summary
            if summary.strip().startswith("{"):
                try:
                    summary_json = json.loads(summary)
                    summary_text = "\n".join(f"- {key}: {value}" for key, value in summary_json.items())
                except json.JSONDecodeError:
                    pass
            details_text += f"Summary:\n{summary_text}\n\n"
        else:
            details_text += "Summary:\nNo details provided.\n\n"
        
        if regions:
            details_text += "Detected regions:\n"
            for r in regions:
                details_text += f"• {r.get('name', 'Unknown')}: {r.get('description', 'N/A')} ({r.get('confidence', 'unknown')} confidence)\n"
        else:
            details_text += "\nNo specific altered regions detected."
            
        self.details_textbox.configure(state="normal")
        self.details_textbox.delete("0.0", "end")
        self.details_textbox.insert("0.0", details_text)
        self.details_textbox.configure(state="disabled")
        
        # Show analyzed image
        if self.analyzed_image:
            self.display_result_image(self.analyzed_image)
        else:
            self.result_image_label.configure(image=None, text="Analysis results will appear here")
                
    def display_result_image(self, img):
        """Display an image in the result panel"""
        display_size = (420, 420)
        img_display = img.copy()
        img_display.thumbnail(display_size, Image.Resampling.LANCZOS)
        
        photo = ctk.CTkImage(light_image=img_display, dark_image=img_display, size=img_display.size)
        self.result_image_label.configure(image=photo, text="")
        self.result_image_label.image = photo
        
    def show_api_key_dialog(self):
        """Show dialog to enter API key"""
        self.loading_frame.place_forget()
        self.start_button.configure(state="normal")
        
        dialog = ctk.CTkInputDialog(
            text="Enter your Gemini API Key:",
            title="API Key Required"
        )
        api_key = dialog.get_input()
        
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            self.start_analysis()
        else:
            messagebox.showwarning("API Key Required", 
                "Please set GEMINI_API_KEY environment variable or enter it when prompted.")
            
    def show_error(self, message):
        """Show error message"""
        self.loading_frame.place_forget()
        self.start_button.configure(state="normal")
        messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{message}")


def main():
    app = IFDIDApp()
    app.mainloop()


if __name__ == "__main__":
    main()

