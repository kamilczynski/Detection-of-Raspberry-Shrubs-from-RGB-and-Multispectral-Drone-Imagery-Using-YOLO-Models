import os
import cv2
import tifffile as tiff
import inspect
import subprocess
import customtkinter as ctk
from tkinter import filedialog, messagebox

# ----------------------------------------------------------------------
# 1. TIFF ➜ scaling ➜ metadata preservation
# ----------------------------------------------------------------------
# (Functions unchanged from the original script)
def resize_tif_with_metadata(src_path: str, dst_path: str,
                             width: int, height: int) -> None:
    with tiff.TiffFile(src_path) as tf:
        page = tf.pages[0]
        img = page.asarray()

        extratags = []
        for tag in page.tags.values():
            if tag.code in (256, 257):
                continue
            extratags.append((tag.code, tag.dtype, tag.count, tag.value, True))

        extratags += [
            (256, 'I', 1, width, True),
            (257, 'I', 1, height, True),
        ]

        write_kwargs = dict(
            photometric=page.photometric,
            bitspersample=page.bitspersample,
            resolution=page.tags.get('XResolution', None),
            extratags=extratags,
            description=page.description
        )
        if 'sampleformat' in inspect.signature(tiff.TiffWriter.write).parameters:
            write_kwargs['sampleformat'] = getattr(page, 'sampleformat', None)

        bigtiff_flag = tf.is_bigtiff

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    with tiff.TiffWriter(dst_path, bigtiff=bigtiff_flag) as tw:
        tw.write(img, **write_kwargs)


def copy_exif_gps(src_path: str, dst_path: str) -> None:
    try:
        subprocess.run([
            "exiftool", "-q", "-q", "-overwrite_original",
            "-TagsFromFile", src_path,
            "-ExifIFD:all", "-GPS*:all", "-MakerNotes:all", "-unsafe",
            dst_path
        ], check=True)
    except FileNotFoundError:
        print("⚠️ ExifTool not found – Exif/GPS data were not copied.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ ExifTool returned an error: {e}")


def resize_images(input_folder: str, output_folder: str,
                  width: int, height: int) -> None:
    os.makedirs(output_folder, exist_ok=True)
    for filename in os.listdir(input_folder):
        src = os.path.join(input_folder, filename)
        dst = os.path.join(output_folder, filename)
        low = filename.lower()
        try:
            if low.endswith(('.tif', '.tiff')):
                resize_tif_with_metadata(src, dst, width, height)
                copy_exif_gps(src, dst)
            elif low.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"⚠️ Failed to load {filename}")
                    continue
                img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst, img)
            else:
                continue
            print(f"Rescaled + metadata preserved: {filename}")
        except Exception as e:
            print(f"❌ Error processing file {filename}: {e}")
    messagebox.showinfo("Done", "Scaling images completed!")

# ----------------------------------------------------------------------
# 4. GUI (Minimalistic, dark background, modern style)
# ----------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk(fg_color="black")
app.title("SpectralPiranhaPix")
app.geometry("600x300")
app.resizable(False, False)

FONT = ("Orbitron", 14)

# Helper functions

def select_input_folder():
    path = filedialog.askdirectory(title="Select input folder")
    input_entry.delete(0, ctk.END)
    input_entry.insert(0, path)


def select_output_folder():
    path = filedialog.askdirectory(title="Select output folder")
    output_entry.delete(0, ctk.END)
    output_entry.insert(0, path)


def start_processing():
    inp = input_entry.get()
    out = output_entry.get()
    w = width_entry.get()
    h = height_entry.get()

    if not (inp and out and w and h):
        messagebox.showerror("Error", "Please fill in all fields.")
        return
    try:
        resize_images(inp, out, int(w), int(h))
    except ValueError:
        messagebox.showerror("Error", "Dimensions must be integers.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")

# Layout

container = ctk.CTkFrame(app, fg_color="black", corner_radius=0)
container.pack(fill="both", expand=True, padx=20, pady=20)

ctk.CTkLabel(container, text="PiranhaPix", font=("Orbitron", 20), text_color="white").pack(pady=(0, 20))

# Input folder
in_frame = ctk.CTkFrame(container, fg_color="black", corner_radius=0)
in_frame._
