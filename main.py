import slides
import emotion

import concurrent.futures

class Main:
    def __init__(self, presentation_path="./presentation.pdf"):
        self.presentation_path = presentation_path
        

    def start_slides(self):
        s = slides.Slides(self.presentation_path)
        s.convert_slide_to_images() 
        s.start()
        s.log_presentation_description()

    def start_emotion(self):
        e = emotion.Emotion()
        e.main()

    def main(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(self.start_slides)
            executor.submit(self.start_emotion)

if __name__ == "__main__":
    main = Main()
    main.main()