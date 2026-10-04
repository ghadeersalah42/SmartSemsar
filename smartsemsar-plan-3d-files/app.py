"""
Smart Semsar - demo page for the furniture pipeline (Gradio).
صفحة تجربة: اختار إعلان، حدد طلباتك، وشوف الفرش في جولة 3D.

Same pipeline as inference.py, with a form instead of command-line flags.

    python app.py            -> http://127.0.0.1:7860
"""
import time
from pathlib import Path

import gradio as gr

from backend.schema.design import DesignPreferences
from backend.schema.staging import load_catalog
from backend.services import vision_service
from inference import OUT_DIR, PipelineError, load_listings, run

OUT_DIR.mkdir(exist_ok=True)
gr.set_static_paths(paths=[OUT_DIR])        # lets the page load the walkthrough files it builds

LISTINGS = load_listings()
LISTING_CHOICES = [(f"{r.property_id} · {str(r.title).strip()[:55]} · {r.pf_area_sqm:.0f} m² · {r.pf_bedrooms} bed",
                    r.property_id) for r in LISTINGS.itertuples()]
KINDS = sorted({i.kind for i in load_catalog().items})
AUTO = "Auto"
EMPTY_VIEW = "<div style='padding:40px;text-align:center;color:#888'>The 3D walkthrough appears here.</div>"


def read_photo(photo, style_brief):
    """Vision tool on upload: fills "What is it?" and the width, or says why the photo will not work."""
    same = gr.update()
    if not photo:
        return same, same, "", same
    if not vision_service.is_configured():
        return same, same, f"No `{vision_service.KEY_ENV}` set, so choose what it is yourself.", same
    try:
        seen = vision_service.analyze_photo(photo)
    except vision_service.VisionError as e:
        return same, same, f"Could not read the photo: {e}", same
    brief = seen.description if seen.description and not (style_brief or "").strip() else same
    if not seen.usable:
        return same, same, f"**This photo cannot be turned into 3D.** {seen.reason}", brief
    width = f", about {seen.width_m} m wide" if seen.width_m else ""
    return seen.kind, seen.width_m, f"Recognised **{seen.name or seen.kind}** as `{seen.kind}`{width}.", brief


def furnish(property_id, density, must_have, exclude, dining_seats, style_brief,
            photo, model, kind, width_m, yaw_deg):
    prefs = DesignPreferences.from_loose({
        "density": density, "must_have": must_have or [], "exclude": exclude or [],
        "dining_seats": None if dining_seats == AUTO else int(dining_seats),
        "style_brief": (style_brief or "").strip() or None,
    })
    lines: list[str] = []
    try:
        html = run(property_id, prefs, photo=photo, model=model, kind=kind or None,
                   width_m=width_m or None, yaw_deg=float(yaw_deg), say=lines.append)
    except PipelineError as e:
        return EMPTY_VIEW, "\n".join(lines + [f"error: {e}"])
    if html is None:
        return EMPTY_VIEW, "\n".join(lines)
    # the page is a file under out/, served by Gradio; the timestamp defeats the browser cache
    src = f"/gradio_api/file={Path(html).resolve().as_posix()}?t={int(time.time())}"
    frame = (f'<iframe src="{src}" title="3D walkthrough" allow="fullscreen" allowfullscreen '
             f'style="width:100%;height:680px;border:0;border-radius:8px"></iframe>')
    return frame, "\n".join(lines)


with gr.Blocks(title="Smart Semsar - furnish a listing") as demo:
    gr.Markdown("# Smart Semsar — furnish a listing\n"
                "Pick a listing, say what you want, and walk through the furnished 3D plan.")
    with gr.Row():
        with gr.Column(scale=1, min_width=330):
            property_id = gr.Dropdown(LISTING_CHOICES, value="PROP_1002", label="Listing")
            density = gr.Radio(["minimal", "normal", "full"], value="normal", label="How much furniture")
            must_have = gr.CheckboxGroup(KINDS, label="Must have")
            exclude = gr.CheckboxGroup(KINDS, label="Leave out")
            dining_seats = gr.Radio([AUTO, "4", "6"], value=AUTO, label="Dining seats")
            style_brief = gr.Textbox(label="Style description (saved with the result, not used for placement yet)",
                                     placeholder="e.g. calm modern, light wood, grey fabric")
            with gr.Accordion("Use my own furniture", open=False):
                gr.Markdown("A photo of **one** piece needs the Colab server running. "
                            "A `.glb` model works without it.")
                photo = gr.Image(type="filepath", label="Photo of one piece")
                photo_note = gr.Markdown()
                model = gr.File(file_types=[".glb"], type="filepath", label="…or a GLB model")
                kind = gr.Dropdown(KINDS, label="What is it? (filled from the photo when possible)")
                width_m = gr.Number(label="Real width in meters (empty = same as the stock item)", value=None)
                yaw_deg = gr.Radio(["0", "90", "180", "270"], value="0", label="Turn it (if it faces sideways)")
            go = gr.Button("Furnish", variant="primary")
        with gr.Column(scale=3):
            view = gr.HTML(EMPTY_VIEW)
            report = gr.Textbox(label="What happened", lines=12, max_lines=20)

    inputs = [property_id, density, must_have, exclude, dining_seats, style_brief, photo, model, kind, width_m, yaw_deg]
    photo.change(read_photo, [photo, style_brief], [kind, width_m, photo_note, style_brief])
    go.click(furnish, inputs, [view, report])
    demo.load(furnish, inputs, [view, report])      # show the default listing straight away

if __name__ == "__main__":
    demo.launch()
