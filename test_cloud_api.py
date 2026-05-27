import os
from gradio_client import Client, handle_file

def test_idm_vton():
    print("Connecting to IDM-VTON Cloud API...")
    client = Client("yisol/IDM-VTON")
    
    person_path = "database/data/tryon_debug/debug_person.png"
    garment_path = "database/data/garments/shirt-001/garment.png"
    
    if not os.path.exists(person_path) or not os.path.exists(garment_path):
        print("Test images not found.")
        return

    # Image editor dict format for Gradio
    dict_img = {
        "background": handle_file(person_path),
        "layers": [],
        "composite": None
    }
    
    print("Sending request... (this may take a few seconds on ZeroGPU)")
    try:
        result = client.predict(
            dict=dict_img,
            garm_img=handle_file(garment_path),
            garment_des="A cool shirt",
            is_checked=True,
            is_checked_crop=False,
            denoise_steps=30,
            seed=42,
            api_name="/tryon"
        )
        print("Success! Output saved at:", result)
    except Exception as e:
        print("API Error:", e)

if __name__ == "__main__":
    test_idm_vton()
