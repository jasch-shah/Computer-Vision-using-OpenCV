import cv2
import pysrt
import argparse

def mid(x,y):
	return (x+y)/2

def save_new_frame(file, ms, text):
	
	videocapture = cv2.VideoCapture(videofile)
	videocapture.set(cv2.CAP_PROP_POS_MSEC,ms)
	success, image = videocapture.read()

	height, width = image.shape[:2]

	if success:
	        font = cv2.FONT_HERSHEY_SIMPLEX

	        overlay = image.copy()

	        cv2.rectangle(overlay, (1, height-15-11), (width-1, height-1), (255,0,0), -1)

	        opacity = 0.4

	        cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, image)

	        cv2.putText(image,text,(10,height-10), font, 0.4, (255,255,255),1)
	        cv2.imwrite('frame%04d.jpg' % (ms/1000), image)



if __name__ == '__main__':
	   parser = argparse.ArgumentParser()

	   parser.add_argument('video', help = 'path to video file')
	   parser.add_argument('sub', help = 'path to sub file')

	   args = parser.parse_args()
	   subs = pysrt.open(args.sub)
	   videofile = args.video

	   for i in xrange(0, len(subs)):
	           seconds_start = (subs[i].start.seconds) + (subs[i].start.minutes * 60) + (subs[i].start.hours * 60 * 60)
	           seconds_end = (subs[i].start.seconds) + (subs[i].start.minutes * 60) + (subs[i].start.hours * 60 * 60)

	           ms = mid(seconds_start*1000,seconds_end*1000)
	           save_new_frame(videofile, ms, str(subs[i].text))       	