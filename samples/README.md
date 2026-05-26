# Example MRI Images

This folder is where you should place example MRI images that the Gradio demo
can showcase. The app will automatically pick them up at startup.

## Adding examples

Drop any `.png`, `.jpg`, or `.jpeg` files into this folder. Recommended:

* Include at least one example per class (glioma, meningioma, pituitary, no tumor)
* Use the preprocessed PNGs from your training pipeline so they match the
  distribution the model was trained on
* Keep filenames descriptive — e.g. `example_01_glioma.png`

The app reads them via `os.listdir()` at launch and exposes them in the
"Examples" section of the Gradio interface.

## Note

Sample images bundled with the repository are reproduced from the original
public datasets (Figshare and Br35H). They are included strictly for
demonstration purposes and remain governed by the licences of their
original sources.
