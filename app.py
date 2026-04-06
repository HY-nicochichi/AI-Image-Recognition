from io import BytesIO
from PIL.Image import (
    Image,
    open
)
from torch import (
    Tensor,
    device,
    no_grad,
    max
)
from torchvision.models import (
    ResNet,
    ResNet50_Weights,
    resnet50
)
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize
)
from uvicorn import run
from pydantic import BaseModel
from fastapi import (
    FastAPI,
    UploadFile
)

cpu_device = device('cpu')

image_formatter = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

identify_model: ResNet = resnet50(weights=ResNet50_Weights.DEFAULT)
identify_model.to(cpu_device)
identify_model.eval()

def identify_object(image: Image) -> str:
    image_tensor: Tensor = image_formatter(image)
    with no_grad():
        probabilities: Tensor = identify_model(image_tensor.unsqueeze(0))
    _, predicted_object = max(probabilities, 1)
    return ResNet50_Weights.DEFAULT.meta['categories'][predicted_object.item()]

app = FastAPI(title='AI Image Recognition API', version='1.0.0')

class IdentifyResponse(BaseModel):
    result: str

@app.post('/identify', tags=['Image Recognition'], summary='Image Recognition')
async def identify(image_file: UploadFile) -> IdentifyResponse:
    image_byte_data: bytes = await image_file.read()
    image: Image = open(BytesIO(image_byte_data)).convert('RGB')
    result: str = identify_object(image)
    return IdentifyResponse(result=result)

if __name__ == '__main__':
    run(app, host='0.0.0.0', port=8000)
