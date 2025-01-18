from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from pprint import pprint
from ultralytics import YOLO
import uuid
import cv2
def generate_id() -> str:
    return uuid.uuid4()
class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def setup(self):
        """Configure any parameters of your model here
        """
        self.set("model_version", "YOLO")
        self.model = YOLO('/home/liujiawei/ultralytics-main/runs/pose/train30/weights/best.pt', task='predict')
        self.rectanglelabels = ['装甲板']
        self.keypointslabels  = ['左上', '左中', '左下', '右上', '右中', '右下']

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        pprint(tasks[0])
        image_path = '/'+tasks[0]['data']['img'][21:]
        image = cv2.imread(image_path)
        # cv2.imshow("frameo", image)
        result = self.model(image_path)[0]
        # print(result)
        clss = result.boxes.cls.tolist()
        conf = result.boxes.conf.tolist()
        boxes = result.boxes.xywhn.tolist()
        keypoints = result.keypoints.xyn.tolist()
        predict_results = list(zip(clss, conf, boxes,keypoints))
        # print(predict_results)
        result = []
        for r in predict_results:
            if r[0] == 0:

                print(r[3])
                print(enumerate(r[3]))
               
                for index, center in enumerate(r[3]):
                 
                    result.append({
                    'from_name': 'kp-1',
                    'id': generate_id(),
                    'to_name': 'img-1',
                    'type': 'keypointlabels',
                    'value': {'keypointlabels': [self.keypointslabels[index]],
                            'width': 0.19230769230769232,
                            'x': 100 * center[0],
                            'y': 100 * center[1]
                            }})
            result.append({
                    "id": generate_id(),
                    "from_name": "label",
                    "to_name": "img-1",
                    "type": "rectanglelabels",
                    "value": {
                        "rectanglelabels": [ self.rectanglelabels[int(r[0])] ],
                        "x": (r[2][0] - r[2][2] / 2) * 100,
                        "y": (r[2][1] - r[2][3] / 2) * 100,
                        "width": r[2][2] * 100,
                        "height": r[2][3] * 100,
                        "rotation": 0
                    }
            })
        predictions = [{
            "model_version": self.get("model_version"),
            "score": 1,
            "task": tasks[0]['id'],
            "updated_at": tasks[0]['updated_at'],
            "result": result
        }]
        pprint(result)
        # example for resource downloading from Label Studio instance,
        # you need to set env vars LABEL_STUDIO_URL and LABEL_STUDIO_API_KEY
        # path = self.get_local_path(tasks[0]['data']['image_url'], task_id=tasks[0]['id'])
        # example for simple classification
        # return [{
        #     "model_version": self.get("model_version"),
        #     "score": 0.12,
        #     "result": [{
        #         "id": "vgzE336-a8",
        #         "from_name": "sentiment",
        #         "to_name": "text",
        #         "type": "choices",
        #         "value": {
        #             "choices": [ "Negative" ]
        #         }
        #     }]
        # }]
        return ModelResponse(predictions=predictions)
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """
        # use cache to retrieve the data from the previous fit() runs
        print('fit() completed successfully.')