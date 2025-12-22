from ultralytics import FastSAM
import cv2
import os

model = FastSAM("FastSAM-x.pt")
results = model.predict(source=os.path.relpath("download.jpg"), conf=0.5, device=0)

for result in results:
    print(result.names)
    # annotated_frame = result.plot()
    # cv2.imshow("Annotated Frame", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()