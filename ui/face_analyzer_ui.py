import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import json
import face_recognition
from tkinter import messagebox, simpledialog
import os
import datetime

class FaceAnalyzerUI:
    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.setup_main_window()
        self.setup_menu()
        
    def setup_main_window(self):
        """Initialize main window and frames"""
        self.root = tk.Tk()
        self.root.title("Face Analysis System")
        
        # Create main container
        self.main_container = tk.Frame(self.root)
        self.main_container.pack(expand=True, fill=tk.BOTH)
        
        # Create camera frame
        self.camera_frame = tk.Frame(self.main_container)
        self.camera_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()
        
        # Create stats frame with same height as camera
        self.stats_frame = None
        self.show_stats = False
        
        # Add scrollable stats frame
        self.stats_canvas = None
        self.stats_scroll = None
        
        # Bind keyboard events
        self.root.bind('q', lambda e: self.analyzer.cleanup())
        self.root.bind('l', lambda e: self.toggle_stats())
        self.root.bind('v', lambda e: self.toggle_voice_commands())
        self.root.bind('g', lambda e: self.toggle_gesture_control())
        
    def setup_menu(self):
        """Create menu bar with options"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # Analysis menu with proper checkmarks
        analysis_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)
        
        # Create BooleanVar for each feature
        self.feature_vars = {}
        features = {
            'mood_tracking': 'Mood Tracking',
            'health_monitoring': 'Health Monitoring',
            'attention_tracking': 'Attention Tracking',
            'security_monitoring': 'Security Monitoring',
            'gesture_control': 'Gesture Control',
            'voice_commands': 'Voice Commands',
            'environment_analysis': 'Environment Analysis',
            'face_recognition_enabled': 'Face Recognition'
        }
        
        # Add menu items with proper config keys
        for config_key, display_name in features.items():
            self.feature_vars[config_key] = tk.BooleanVar(
                value=self.analyzer.config.get(config_key, False))
            analysis_menu.add_checkbutton(
                label=display_name,
                variable=self.feature_vars[config_key],
                command=lambda k=config_key: self.toggle_feature(k)
            )
        
        # Security menu
        security_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Security", menu=security_menu)
        security_menu.add_checkbutton(
            label="Face Recognition",
            variable=tk.BooleanVar(value=self.analyzer.config['face_recognition_enabled']),
            command=lambda: self.toggle_feature('face_recognition_enabled')
        )
        security_menu.add_command(label="Add Known Face", command=self.add_known_face)
        security_menu.add_command(label="View Known Faces", command=self.view_known_faces)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        settings_menu.add_command(label="Configure Alerts", command=self.configure_alerts)
        settings_menu.add_command(label="System Settings", command=self.system_settings)
        
    def update_camera_frame(self, cv2_image):
        """Update camera frame with new image"""
        image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(image=image)
        self.camera_label.photo = photo
        self.camera_label.configure(image=photo)
        
    def update_stats_frame(self, cv2_stats_image):
        """Update statistics window with scrolling"""
        if self.show_stats:
            if self.stats_frame is None:
                # Create stats frame aligned to the right of camera
                self.stats_frame = tk.Frame(self.main_container)
                self.stats_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
                
                # Get camera frame height for alignment
                camera_height = self.camera_label.winfo_height()
                
                # Create canvas with fixed width and matching height
                self.stats_canvas = tk.Canvas(self.stats_frame, 
                                            width=800, 
                                            height=camera_height)
                self.stats_scroll = tk.Scrollbar(self.stats_frame, 
                                               orient="vertical", 
                                               command=self.stats_canvas.yview)
                
                # Configure canvas and scrollbar
                self.stats_canvas.configure(yscrollcommand=self.stats_scroll.set)
                self.stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                self.stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                
                # Create content frame
                self.stats_content = tk.Frame(self.stats_canvas)
                self.stats_label = tk.Label(self.stats_content)
                self.stats_label.pack(expand=True, fill=tk.BOTH)
                
                # Add content window to canvas with proper width
                self.stats_window = self.stats_canvas.create_window(
                    (0, 0),
                    window=self.stats_content,
                    anchor='nw',
                    width=780,
                    height=2000  # Increased height to ensure full content is visible
                )
                
                # Configure scrolling
                self.stats_content.bind('<Configure>', self._configure_stats_scroll)
                self.stats_canvas.bind('<Configure>', self._configure_stats_window)
                
                # Enable mouse wheel scrolling
                self.stats_canvas.bind_all("<MouseWheel>", self._on_mousewheel)
            
            # Update stats image
            stats_image = cv2.cvtColor(cv2_stats_image, cv2.COLOR_BGR2RGB)
            stats_image = Image.fromarray(stats_image)
            stats_photo = ImageTk.PhotoImage(image=stats_image)
            self.stats_label.photo = stats_photo
            self.stats_label.configure(image=stats_photo)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling"""
        self.stats_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _configure_stats_scroll(self, event):
        """Configure stats scrolling region"""
        self.stats_canvas.configure(scrollregion=self.stats_canvas.bbox('all'))
        # Ensure the full content is visible
        content_height = self.stats_content.winfo_reqheight()
        self.stats_canvas.itemconfig(self.stats_window, height=max(content_height, 2000))

    def _configure_stats_window(self, event):
        """Configure stats window size"""
        # Keep width fixed but allow height to expand
        self.stats_canvas.itemconfig(self.stats_window, width=event.width)

    def toggle_stats(self):
        """Toggle statistics window"""
        try:
            self.show_stats = not self.show_stats
            if self.show_stats:
                if self.stats_frame is None:
                    # Create stats frame aligned to the right of camera
                    self.stats_frame = tk.Frame(self.main_container)
                    self.stats_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH)
                    
                    # Get camera frame height
                    camera_height = self.camera_label.winfo_height()
                    
                    # Create scrollable canvas
                    self.stats_canvas = tk.Canvas(self.stats_frame, 
                                               width=800, 
                                               height=camera_height)
                    self.stats_scroll = tk.Scrollbar(self.stats_frame, 
                                                  orient="vertical", 
                                                  command=self.stats_canvas.yview)
                    
                    # Configure scrolling
                    self.stats_canvas.configure(yscrollcommand=self.stats_scroll.set)
                    self.stats_scroll.pack(side=tk.RIGHT, fill=tk.Y)
                    self.stats_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
                    
                    # Create content frame
                    self.stats_content = tk.Frame(self.stats_canvas)
                    self.stats_label = tk.Label(self.stats_content)
                    self.stats_label.pack(expand=True, fill=tk.BOTH)
                    
                    # Add content window to canvas
                    self.stats_window = self.stats_canvas.create_window(
                        (0, 0),
                        window=self.stats_content,
                        anchor='nw',
                        width=780
                    )
                    
                    # Configure canvas scrolling
                    self.stats_content.bind('<Configure>', self._configure_stats_scroll)
                    self.stats_canvas.bind('<Configure>', self._configure_stats_window)
            else:
                if self.stats_frame:
                    self.stats_frame.destroy()
                    self.stats_frame = None
                    self.stats_canvas = None
                    self.stats_scroll = None
                    self.stats_content = None
                    self.stats_label = None
                
        except Exception as e:
            self.analyzer.logger.error(f"[face_analyzer_ui.py] Stats toggle error: {str(e)}")
            self.show_stats = False
            if self.stats_frame:
                self.stats_frame.destroy()
                self.stats_frame = None
                
    def add_known_face(self):
        """Add new face to known faces database"""
        try:
            # Check if we have a frame and analysis
            if self.analyzer.last_frame is None:
                messagebox.showwarning("Warning", "No frame available!")
                return
            
            if not hasattr(self.analyzer, 'current_analysis') or self.analyzer.current_analysis is None:
                messagebox.showwarning("Warning", "Please wait for face analysis to complete!")
                return
            
            # Get name for the face
            name = simpledialog.askstring("Input", "Enter name for this face:")
            if not name:
                return
            
            try:
                # Save current frame
                face_img = self.analyzer.last_frame.copy()
                
                # Ensure directories exist
                os.makedirs('face_detection_output/known_faces', exist_ok=True)
                
                # Save face image
                cv2.imwrite(f'face_detection_output/known_faces/{name}.jpg', face_img)
                
                # Initialize known_faces if needed
                if not hasattr(self.analyzer, 'known_faces'):
                    self.analyzer.known_faces = {}
                
                # Save face data
                self.analyzer.known_faces[name] = {
                    'timestamp': str(datetime.datetime.now()),
                    'analysis': self.analyzer.current_analysis
                }
                
                # Save to JSON
                with open('face_detection_output/known_faces.json', 'w') as f:
                    json.dump(self.analyzer.known_faces, f)
                    
                messagebox.showinfo("Success", f"Face added for {name}")
                
            except Exception as e:
                self.analyzer.logger.error(f"[face_analyzer_ui.py] Failed to save face: {str(e)}")
                messagebox.showerror("Error", "Failed to save face data")
                
        except Exception as e:
            self.analyzer.logger.error(f"[face_analyzer_ui.py] Face addition error: {str(e)}")
            messagebox.showerror("Error", "Failed to process face")

    def view_known_faces(self):
        """View and manage known faces"""
        try:
            # Load known faces from file
            known_faces_file = os.path.join('face_detection_output', 'known_faces.json')
            if not os.path.exists(known_faces_file):
                messagebox.showinfo("Info", "No known faces in database")
                return
            
            with open(known_faces_file, 'r') as f:
                self.analyzer.known_faces = json.load(f)
            
            if not self.analyzer.known_faces:
                messagebox.showinfo("Info", "No known faces in database")
                return
            
            # Create view window
            view_window = tk.Toplevel(self.root)
            view_window.title("Known Faces")
            view_window.geometry("400x600")
            
            # Add scrollable frame
            canvas = tk.Canvas(view_window)
            scrollbar = tk.Scrollbar(view_window, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Add faces to scrollable frame
            face_img_dir = os.path.join('face_detection_output', 'known_faces')
            for name in self.analyzer.known_faces:
                frame = tk.Frame(scrollable_frame)
                frame.pack(pady=5, padx=5, fill=tk.X)
                
                # Try to load and display face image
                img_path = os.path.join(face_img_dir, f"{name}.jpg")
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    img = img.resize((50, 50), Image.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    img_label = tk.Label(frame, image=photo)
                    img_label.image = photo
                    img_label.pack(side=tk.LEFT, padx=5)
                    
                tk.Label(frame, text=name, width=20).pack(side=tk.LEFT, padx=5)
                tk.Button(
                    frame, 
                    text="Remove",
                    command=lambda n=name: self.remove_known_face(n, view_window)
                ).pack(side=tk.RIGHT, padx=5)
            
            # Pack scrollbar and canvas
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
        except Exception as e:
            self.analyzer.logger.error(f"[face_analyzer_ui.py] View faces error: {str(e)}")
            messagebox.showerror("Error", "Failed to load known faces")

    def configure_alerts(self):
        """Configure alert settings"""
        alert_window = tk.Toplevel(self.root)
        alert_window.title("Alert Configuration")
        
        for alert_type in self.analyzer.config['sounds']:
            var = tk.BooleanVar(value=self.analyzer.config['sounds'][alert_type])
            tk.Checkbutton(
                alert_window,
                text=alert_type.replace('_', ' ').title(),
                variable=var,
                command=lambda t=alert_type, v=var: self.toggle_alert(t, v.get())
            ).pack()
            
    def system_settings(self):
        """Configure system settings"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("System Settings")
        
        # FPS settings
        tk.Label(settings_window, text="Target FPS:").pack()
        fps_var = tk.StringVar(value=str(self.analyzer.config['target_fps']))
        tk.Entry(settings_window, textvariable=fps_var).pack()
        
        # Resolution settings
        tk.Label(settings_window, text="Resolution:").pack()
        res_var = tk.StringVar(value=f"{self.analyzer.config['frame_width']}x{self.analyzer.config['frame_height']}")
        tk.Entry(settings_window, textvariable=res_var).pack()
        
        # Save button
        tk.Button(
            settings_window,
            text="Save",
            command=lambda: self.save_system_settings(fps_var.get(), res_var.get())
        ).pack()
        
    def toggle_voice_commands(self):
        """Toggle voice command recognition"""
        self.analyzer.config['enable_voice_commands'] = not self.analyzer.config['enable_voice_commands']
        status = "enabled" if self.analyzer.config['enable_voice_commands'] else "disabled"
        messagebox.showinfo("Voice Commands", f"Voice commands {status}")

    def toggle_gesture_control(self):
        """Toggle gesture control recognition"""
        self.analyzer.config['enable_gesture_control'] = not self.analyzer.config['enable_gesture_control']
        status = "enabled" if self.analyzer.config['enable_gesture_control'] else "disabled"
        messagebox.showinfo("Gesture Control", f"Gesture control {status}")

    def save_system_settings(self, fps, resolution):
        """Save system settings"""
        try:
            self.analyzer.config['target_fps'] = int(fps)
            width, height = map(int, resolution.split('x'))
            self.analyzer.config['frame_width'] = width
            self.analyzer.config['frame_height'] = height
            self.analyzer.cap.set(cv2.CAP_PROP_FPS, int(fps))
            self.analyzer.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.analyzer.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            messagebox.showinfo("Success", "Settings saved successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")

    def toggle_feature(self, feature_key):
        """Toggle feature with proper checkmark update"""
        current_state = self.feature_vars[feature_key].get()
        self.analyzer.config[feature_key] = current_state
        # Update checkmark
        self.feature_vars[feature_key].set(current_state)
        
        status = "enabled" if current_state else "disabled"
        messagebox.showinfo("Feature Toggle", 
                           f"{feature_key.replace('_', ' ').title()} {status}")

    def toggle_alert(self, alert_type, enabled):
        """Toggle alert settings"""
        self.analyzer.config['sounds'][alert_type] = enabled
        status = "enabled" if enabled else "disabled"
        messagebox.showinfo("Alert Settings", f"{alert_type.replace('_', ' ').title()} alerts {status}")

    def remove_known_face(self, name, window=None):
        """Remove face from known faces database"""
        try:
            # Remove from dictionary
            if name in self.analyzer.known_faces:
                del self.analyzer.known_faces[name]
            
            # Update file
            known_faces_file = os.path.join('face_detection_output', 'known_faces.json')
            with open(known_faces_file, 'w') as f:
                json.dump(self.analyzer.known_faces, f)
            
            # Try to remove image file
            img_path = os.path.join('face_detection_output', 'known_faces', f"{name}.jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            
            messagebox.showinfo("Success", f"Removed {name} from database")
            
            # Refresh view window if provided
            if window:
                window.destroy()
                self.view_known_faces()
            
        except Exception as e:
            self.analyzer.logger.error(f"[face_analyzer_ui.py] Remove face error: {str(e)}")
            messagebox.showerror("Error", f"Failed to remove face: {str(e)}")
        
    def run(self):
        """Start the UI main loop"""
        self.root.mainloop() 