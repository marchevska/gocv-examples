// Edit video recorded with ORB detector
//
// On my laptop detection on live webcam input took about 80-90 ms,
// so I inserted transitions between original frames to make the video slower
// while keeping the frame rate, also inserted intro frames with fade in/out transtions

package main

import (
	"errors"
	"fmt"
	"image"
	"image/color"

	"gocv.io/x/gocv"
)

type (
	myVWManager struct {
		vWriter   *gocv.VideoWriter
		lastFrame *gocv.Mat
	}
)

const (
	inputVideo  = "video1.avi"
	outputVideo = "video_edited.avi"
	videoCodec  = "MJPG"
	outputFPS   = 30
	frameType   = gocv.MatTypeCV8UC3
	font        = gocv.FontHersheyTriplex
)

var (
	white    = color.RGBA{255, 255, 255, 0}
	darkblue = color.RGBA{0, 0, 127, 0}
)

func (vwm *myVWManager) RepeatFrame(img *gocv.Mat, delay float64) (err error) {
	if !vwm.vWriter.IsOpened() {
		return errors.New("Cannot write to the file")
	}
	nFrames := int(delay * outputFPS)
	for i := 0; i <= nFrames; i++ {
		vwm.lastFrame = img
		err = vwm.vWriter.Write(*img)
		if err != nil {
			return
		}
	}
	return
}

// FadeImageInto writes a fading in sequence to the video file
// Starting image img1, ending image img2, duration delay milliseconds
func (vwm *myVWManager) FadeImageInto(img1, img2 *gocv.Mat, delay float64) (err error) {
	if delay <= 0 {
		return fmt.Errorf("Cannot make transition of %f seconds", delay)
	}
	if !vwm.vWriter.IsOpened() {
		return errors.New("Cannot write to the file")
	}

	img3 := gocv.NewMat()
	alpha, beta := 0.0, 1.0
	nFrames := int(delay * outputFPS)

	for i := 0; i <= nFrames; i++ {
		if i == nFrames-1 {
			alpha, beta = 0, 1
		} else {
			beta = float64(i) / float64(nFrames-1)
			alpha = 1 - beta
		}
		gocv.AddWeighted(*img1, alpha, *img2, beta, 1, &img3)
		vwm.lastFrame = &img3
		err = vwm.vWriter.Write(img3)
		if err != nil {
			return
		}
	}
	return
}

// CopyFrom copies frames from the video and adds intermediate frames since the original video is slow
func (vwm *myVWManager) CopyFrom(vr *gocv.VideoCapture, delay float64) (err error) {
	if delay <= 0 {
		return fmt.Errorf("Wrong duration specified: %f seconds", delay)
	}
	if !vwm.vWriter.IsOpened() {
		return errors.New("Cannot write to the file")
	}
	if !vwm.vWriter.IsOpened() {
		return errors.New("Cannot write to the file")
	}

	nFrames := int(delay * outputFPS)
	extraFrame := gocv.NewMat()
	img := gocv.NewMat()
	for i := 0; i < nFrames; i++ {
		vr.Read(&img)
		gocv.AddWeighted(*vwm.lastFrame, 0.7, img, 0.3, 1, &extraFrame)
		vwm.vWriter.Write(extraFrame)
		gocv.AddWeighted(*vwm.lastFrame, 0.3, img, 0.7, 1, &extraFrame)
		vwm.vWriter.Write(extraFrame)
		vwm.vWriter.Write(img)
		vwm.lastFrame = &img
	}
	return
}

// MessageBox creates and returns an image with plain background and specified text lines
// Text is center horizontally and vertically; a default font is used
func MessageBox(lines []string, textColor, bgColor color.RGBA, fontScale, lineHeight float64,
	thickness, width, height int) (img gocv.Mat) {
	img = gocv.NewMatWithSize(height, width, frameType)
	gocv.Rectangle(&img, image.Rect(0, 0, width, height), bgColor, -1)

	if len(lines) > 0 {
		textHeightPixels := gocv.GetTextSize(lines[0], font, fontScale, thickness).Y
		lineHeightPixels := int(float64(textHeightPixels) * lineHeight)
		totalTextHeight := lineHeightPixels*(len(lines)-1) + textHeightPixels
		startY := (height-totalTextHeight)/2 + textHeightPixels

		for i, s := range lines {
			lineWidthPixels := gocv.GetTextSize(s, font, fontScale, thickness).X
			gocv.PutText(&img, s, image.Pt((width-lineWidthPixels)/2, startY+i*lineHeightPixels),
				font, fontScale, textColor, thickness)
		}
	}
	return img
}

func main() {
	// Create video reader and writer
	vReader, _ := gocv.OpenVideoCapture(inputVideo)
	videoWidth := int(vReader.Get(gocv.VideoCaptureFrameWidth))
	videoHeight := int(vReader.Get(gocv.VideoCaptureFrameHeight))
	vWriter, _ := gocv.VideoWriterFile(outputVideo, videoCodec, outputFPS, videoWidth, videoHeight, true)
	defer vReader.Close()
	defer vWriter.Close()
	vwm := myVWManager{vWriter: vWriter}

	// Intro screens
	blackScreen := gocv.NewMatWithSize(videoHeight, videoWidth, frameType)
	lines := []string{"OpenCV ORB", "playing cards recognition", "example with gocv"}
	introFrame := MessageBox(lines, white, darkblue, 2, 3, 3, videoWidth, videoHeight)
	lines2 := []string{"Continue demonstration", "with closed", "face and suit signs"}
	introFrame2 := MessageBox(lines2, white, darkblue, 2, 3, 3, videoWidth, videoHeight)

	// First frame of the video is used for transitions
	firstFrame := gocv.NewMat()
	vReader.Read(&firstFrame)

	// Add intro screen with fade in-fade out effects
	vwm.FadeImageInto(&blackScreen, &introFrame, 1.5)
	vwm.RepeatFrame(&introFrame, 2.0)
	vwm.FadeImageInto(&introFrame, &firstFrame, 1.2)

	// Copy slowed fragment video from the original video
	vwm.CopyFrom(vReader, 12.1)

	// Second intro screen and the rest of the original video
	vwm.FadeImageInto(vwm.lastFrame, &introFrame2, 1.5)
	vwm.RepeatFrame(&introFrame2, 2.0)
	vwm.FadeImageInto(&introFrame2, vwm.lastFrame, 1.5)
	vwm.CopyFrom(vReader, 36.0)

}
