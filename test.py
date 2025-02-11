import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set environment variables for OpenCV
opencv_dir = os.path.join(os.path.dirname(__file__), 'opencv')
if not os.path.exists(opencv_dir):
    logging.error(f"OpenCV directory not found: {opencv_dir}")
    raise FileNotFoundError(f"OpenCV directory not found: {opencv_dir}")

os.environ['OPENCV_DIR'] = opencv_dir
os.environ['PATH'] += ';' + os.path.join(opencv_dir, 'bin', 'Release')
sys.path.append(os.path.join(opencv_dir, 'lib', 'python3', 'Release')) 

# Check if the OpenCV libraries are available
opencv_bin_dir = os.path.join(opencv_dir, 'bin', 'Release')
if not os.path.exists(opencv_bin_dir):
    logging.error(f"OpenCV bin directory not found: {opencv_bin_dir}")
    raise FileNotFoundError(f"OpenCV bin directory not found: {opencv_bin_dir}")

# Check if the OpenCV Python bindings are available
opencv_python_dir = os.path.join(opencv_dir, 'lib', 'python3', 'Release')
if not any(fname.startswith('cv2') for fname in os.listdir(opencv_python_dir)):
    logging.error(f"OpenCV Python bindings not found in: {opencv_python_dir}")
    raise FileNotFoundError(f"OpenCV Python bindings not found in: {opencv_python_dir}")

# Ensure the DLLs are in the PATH
os.add_dll_directory(opencv_bin_dir)
logging.info(f"Added {opencv_bin_dir} to DLL directory")

# Log the current PATH
logging.info(f"Current PATH: {os.environ['PATH']}")

# Add the OpenCV Python bindings to sys.path
sys.path.append(opencv_python_dir)
logging.info(f"Added {opencv_python_dir} to sys.path")

# Log the current sys.path
logging.info(f"Current sys.path: {sys.path}")

try:
    import cv2
    logging.info("Successfully imported cv2")
except ImportError as e:
    logging.error(f"Failed to import cv2: {e}")
    raise

import tkinter as tk
from tkinter import scrolledtext, font, messagebox
import torch
import torchvision
import threading
import time
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

class CudaTestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Test App")
        self.root.geometry("300x650")

        self.monitor_thread_running = False

        self.set_icon()
        self.create_widgets()

    def set_icon(self):
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
                icon_path = os.path.join(base_path, "icon.ico")
            else:
                icon_path = "icon.ico"

            self.root.iconbitmap(icon_path)
            logging.info("Icon set successfully.")
        except Exception as e:
            logging.error(f"Could not set icon: {e}")
            messagebox.showerror("Error", f"Could not set icon: {e}")

    def create_widgets(self):
        logging.info("Creating widgets...")
        log_font = font.Font(family="Courier", size=10)
        self.log_box = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, width=80, height=25, font=log_font)
        self.log_box.pack(pady=10)

        # Define tags for colored text
        self.log_box.tag_config('true', foreground='green')
        self.log_box.tag_config('false', foreground='red')

        self.display_cuda_info()

        self.start_monitor_button = tk.Button(self.root, text="Start GPU Monitor", command=self.start_gpu_monitor)
        self.start_monitor_button.pack(pady=10)

        self.run_pytorch_test_button = tk.Button(self.root, text="Run PyTorch Test", command=self.run_pytorch_test)
        self.run_pytorch_test_button.pack(pady=10)

        self.run_opencv_test_button = tk.Button(self.root, text="Run OpenCV Test", command=self.run_opencv_test)
        self.run_opencv_test_button.pack(pady=10)

        self.run_torchvision_test_button = tk.Button(self.root, text="Run TorchVision Test", command=self.run_torchvision_test)
        self.run_torchvision_test_button.pack(pady=10)

    def display_cuda_info(self):
        logging.info("Displaying CUDA info...")
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        cudnn_version = torch.backends.cudnn.version() if cuda_available else "N/A"
        
        info = f"CUDA Available: {cuda_available}\nCUDA Version: {cuda_version}\ncuDNN Version: {cudnn_version}\n"
        self.log_box.insert(tk.END, info)

        # Check PyTorch CUDA support and version
        pytorch_cuda = torch.cuda.is_available()
        pytorch_version = torch.__version__
        self.log_box.insert(tk.END, f"PyTorch Version: {pytorch_version}\n")
        self.log_box.insert(tk.END, f"PyTorch CUDA Support: ")
        self.log_box.insert(tk.END, f"{pytorch_cuda}\n", 'true' if pytorch_cuda else 'false')

        # Check TorchVision CUDA support and version
        torchvision_version = torchvision.__version__
        self.log_box.insert(tk.END, f"TorchVision Version: {torchvision_version}\n")
        try:
            dummy_tensor = torch.randn(1, 3, 224, 224).cuda()
            torchvision_cuda = True
        except Exception as e:
            torchvision_cuda = False
        self.log_box.insert(tk.END, f"TorchVision CUDA Support: ")
        self.log_box.insert(tk.END, f"{torchvision_cuda}\n", 'true' if torchvision_cuda else 'false')

        # Check OpenCV CUDA support and version
        try:
            opencv_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
            opencv_version = cv2.__version__
            self.log_box.insert(tk.END, f"OpenCV Version: {opencv_version}\n")
            self.log_box.insert(tk.END, f"OpenCV CUDA Support: ")
            self.log_box.insert(tk.END, f"{opencv_cuda}\n", 'true' if opencv_cuda else 'false')
        except Exception as e:
            self.log_box.insert(tk.END, f"OpenCV CUDA Support check failed: {e}\n")

        logging.info("CUDA info displayed.")

    def start_gpu_monitor(self):
        if self.monitor_thread_running:
            logging.warning("GPU monitor is already running.")
            return

        self.monitor_thread_running = True
        monitor_window = tk.Toplevel(self.root)
        monitor_window.title("GPU Monitor")
        monitor_window.geometry("800x600")

        fig, ax = plt.subplots()
        canvas = FigureCanvasTkAgg(fig, master=monitor_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        stats_text = scrolledtext.ScrolledText(monitor_window, wrap=tk.WORD, width=80, height=10, font=font.Font(family="Courier", size=10))
        stats_text.pack(pady=10)

        gpu_util_data = []

        def update_gpu_stats(i):
            try:
                result = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
                lines = result.stdout.splitlines()
                stats_text.delete("1.0", tk.END)
                for line in lines:
                    gpu_util, gpu_temp, mem_used, mem_total = line.split(", ")
                    gpu_util_data.append(int(gpu_util))
                    stats_text.insert(tk.END, f"GPU Utilization: {gpu_util}%\n")
                    stats_text.insert(tk.END, f"GPU Temperature: {gpu_temp}C\n")
                    stats_text.insert(tk.END, f"Memory Used: {mem_used} MiB\n")
                    stats_text.insert(tk.END, f"Memory Total: {mem_total} MiB\n")
                    stats_text.insert(tk.END, "\n")

                ax.clear()
                ax.plot(gpu_util_data, label='GPU Utilization (%)')
                ax.legend(loc='upper left')
                ax.set_title('GPU Utilization')
                ax.set_xlabel('Time')
                ax.set_ylabel('Utilization (%)')
                canvas.draw()
                logging.info("GPU stats updated.")
            except Exception as e:
                stats_text.insert(tk.END, f"Error retrieving GPU data: {e}\n")
                logging.error(f"Error updating GPU stats: {e}")

        self.ani = FuncAnimation(fig, update_gpu_stats, interval=1000)
        monitor_window.protocol("WM_DELETE_WINDOW", lambda: self.stop_gpu_monitor_thread(monitor_window))
        logging.info("GPU monitor started.")

    def stop_gpu_monitor_thread(self, window):
        logging.info("Stopping GPU monitor...")
        self.monitor_thread_running = False
        window.destroy()
        logging.info("GPU monitor stopped.")

    def run_pytorch_test(self):
        if not torch.cuda.is_available():
            self.log_box.insert(tk.END, "PyTorch CUDA is not available.\n", 'false')
            logging.warning("PyTorch CUDA is not available.")
            return

        logging.info("Running PyTorch test...")
        self.log_box.insert(tk.END, "Running PyTorch test...\n")
        self.log_box.yview(tk.END)

        try:
            start_time = time.time()
            x = torch.rand(10000, 10000).cuda()
            y = torch.matmul(x, x)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.log_box.insert(tk.END, f"PyTorch test passed: {elapsed_time} seconds\n")
            logging.info("PyTorch test passed.")
        except Exception as e:
            self.log_box.insert(tk.END, f"PyTorch test failed: {e}\n")
            logging.error(f"PyTorch test failed: {e}")

    def run_opencv_test(self):
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() == 0:
                self.log_box.insert(tk.END, "OpenCV CUDA is not available.\n", 'false')
                logging.warning("OpenCV CUDA is not available.")
                return
        except Exception as e:
            self.log_box.insert(tk.END, f"OpenCV CUDA Support check failed: {e}\n")
            logging.error(f"OpenCV CUDA Support check failed: {e}")
            return

        logging.info("Running OpenCV test...")
        self.log_box.insert(tk.END, "Running OpenCV test...\n")
        self.log_box.yview(tk.END)

        try:
            start_time = time.time()
            img = cv2.imread('test_image.jpg', cv2.IMREAD_COLOR)
            gpu_mat = cv2.cuda_GpuMat()
            gpu_mat.upload(img)
            gray = cv2.cuda.cvtColor(gpu_mat, cv2.COLOR_BGR2GRAY)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.log_box.insert(tk.END, f"OpenCV test passed: {elapsed_time} seconds\n")
            logging.info("OpenCV test passed.")
        except Exception as e:
            self.log_box.insert(tk.END, f"OpenCV test failed: {e}\n")
            logging.error(f"OpenCV test failed: {e}")

    def run_torchvision_test(self):
        if not torch.cuda.is_available():
            self.log_box.insert(tk.END, "TorchVision CUDA is not available.\n", 'false')
            logging.warning("TorchVision CUDA is not available.")
            return

        logging.info("Running TorchVision test...")
        self.log_box.insert(tk.END, "Running TorchVision test...\n")
        self.log_box.yview(tk.END)

        try:
            start_time = time.time()
            model = torchvision.models.resnet50(weights=None).cuda()
            dummy_input = torch.randn(1, 3, 224, 224).cuda()
            output = model(dummy_input)
            end_time = time.time()
            elapsed_time = end_time - start_time
            self.log_box.insert(tk.END, f"TorchVision test passed: {elapsed_time} seconds\n")
            logging.info("TorchVision test passed.")
        except Exception as e:
            self.log_box.insert(tk.END, f"TorchVision test failed: {e}\n")
            logging.error(f"TorchVision test failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CudaTestApp(root)
    root.mainloop()
