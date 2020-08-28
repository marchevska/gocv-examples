// This example shows how to use Yolo4 with GoCV
// with default settings, to classify objects on a single image
//
// For more information on Darknet and Yolo 4 please visit
// https://github.com/AlexeyAB/darknet
//
// Before using this example, you need to download list model files:
// List of labels: https://github.com/AlexeyAB/darknet/blob/master/cfg/coco.names
// Config file:    https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
// Model weights:  https://drive.google.com/open?id=1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT
//

package main

import (
	"bufio"
	"fmt"
	"image"
	"image/color"
	"log"
	"os"
	"sort"

	"gocv.io/x/gocv"
)

const (
	confThr         = 0.5 // Detection confidence threshold
	ovrThr          = 0.4 // Overlapping threshold for NMS
	blobSize        = 416
	blobScale       = 1.0 / 255        // Value required for Yolo
	imgPath         = "img/person.jpg" // Image for detection
	classLabelsPath = "coco.names"     // Labels list
	yoloConfigPath  = "yolov4.cfg"     // Config file
	yoloWeightsPath = "yolov4.weights" // Model weights
)

const (
	fontFace      = gocv.FontHersheySimplex
	fontScale     = 0.6
	fontThickness = 1
	bboxThickness = 1
	textPadding   = 3
)

var (
	green    = color.RGBA{0, 255, 0, 0}
	darkblue = color.RGBA{0, 0, 127, 0}
	white    = color.RGBA{255, 255, 255, 0}
)

// YoloDetection struct stores single detection information
type YoloDetection struct {
	detClass int
	detName  string
	detConf  float32
	detBBox  image.Rectangle
}

func (d YoloDetection) String() string {
	return fmt.Sprintf("Detected %d: %s, Confidence: %.2f%%, Bbox: %v", d.detClass, d.detName, d.detConf*100, d.detBBox)
}

// YoloDSlice stores a sortable slice of detections
type YoloDSlice []YoloDetection

func (yd YoloDSlice) Len() int           { return len(yd) }
func (yd YoloDSlice) Less(i, j int) bool { return yd[i].detConf < yd[j].detConf }
func (yd YoloDSlice) Swap(i, j int)      { yd[i], yd[j] = yd[j], yd[i] }

func readClassLabels(filename string) (cl []string) {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		cl = append(cl, scanner.Text())
	}
	return
}

// Extract predictions from Yolo output layers
func extractPredictions(detLayers []gocv.Mat, imgSize []int, classLabels []string) YoloDSlice {
	var yd, ydFiltered YoloDSlice
	frameWidth, frameHeight := imgSize[1], imgSize[0]

	for _, prob := range detLayers {
		for j := 0; j < prob.Rows(); j++ {
			row := prob.RowRange(j, j+1)
			scores := row.ColRange(5, prob.Cols())
			_, confidence, _, maxLoc := gocv.MinMaxLoc(scores)
			if confidence > confThr {
				classID := maxLoc.X
				className := classLabels[classID]
				centerX := int(row.GetFloatAt(0, 0) * float32(frameWidth))
				centerY := int(row.GetFloatAt(0, 1) * float32(frameHeight))
				width := int(row.GetFloatAt(0, 2) * float32(frameWidth))
				height := int(row.GetFloatAt(0, 3) * float32(frameHeight))
				left := int(centerX - width/2)
				top := int(centerY - height/2)
				yd = append(yd, YoloDetection{classID, className, confidence,
					image.Rect(left, top, left+width, top+height)})
			}
		}
	}

	// Apply NMS (at the moment of writing, GoCV does not include implementation of NMSBoxes)
	sort.Sort(sort.Reverse(yd))
	for _, d := range yd {
		keep := true
		area := d.detBBox.Size().X * d.detBBox.Size().Y
		for _, df := range ydFiltered {
			overlap := d.detBBox.Intersect(df.detBBox)
			ovArea := overlap.Size().X * overlap.Size().Y
			keep = keep && (float64(ovArea) <= ovrThr*float64(area))
			if !keep {
				break
			}
		}
		if keep {
			ydFiltered = append(ydFiltered, d)
		}
	}

	return ydFiltered
}

// Draw predictions over the image
func drawPredictions(img gocv.Mat, yd YoloDSlice) {
	for _, d := range yd {
		textSize := gocv.GetTextSize(d.detName, fontFace, fontScale, fontThickness)
		bboxMin := d.detBBox.Min
		gocv.Rectangle(&img, image.Rect(bboxMin.X, bboxMin.Y, bboxMin.X+textSize.X+2*textPadding, bboxMin.Y-textSize.Y-2*textPadding),
			darkblue, -1)
		gocv.PutText(&img, d.detName, image.Pt(d.detBBox.Min.X+textPadding, d.detBBox.Min.Y-2*textPadding),
			fontFace, fontScale, white, fontThickness)
		gocv.Rectangle(&img, d.detBBox, green, bboxThickness)
	}
	return
}

func main() {
	// Initialize model
	classLabels := readClassLabels(classLabelsPath)
	yoloModel := gocv.ReadNet(yoloWeightsPath, yoloConfigPath)
	if yoloModel.Empty() {
		fmt.Println("Error loading model")
		return
	}

	// Find names of the layers with type "Region" which are output layers
	// GetLayer argument (layer number) is starting from 1 since layer 0 is "_input"
	// In Yolo 4 configuration, these shoudl be [yolo_139 yolo_150 yolo_161]
	var yoloOutputLayers []string
	yoloLayers := yoloModel.GetLayerNames()
	for i := 0; i < len(yoloLayers); i++ {
		l := yoloModel.GetLayer(i + 1)
		if l.GetType() == "Region" {
			yoloOutputLayers = append(yoloOutputLayers, l.GetName())
		}
	}

	// Read the image and feed it to the netwotk
	img := gocv.IMRead(imgPath, gocv.IMReadColor)
	img2 := img.Clone() // A copy used to create blob and perform detection

	// Image conversion is required to create a blob as explained in
	// https://github.com/hybridgroup/gocv/issues/658
	img2.ConvertTo(&img2, gocv.MatTypeCV32F)
	blob := gocv.BlobFromImage(img2, blobScale, image.Pt(blobSize, blobSize), gocv.NewScalar(0, 0, 0, 0), true, false)
	yoloModel.SetInput(blob, "")

	// Get model output
	// Yolo4 has 3 detection layers, need to forward to each one separately
	var detLayers []gocv.Mat
	for _, l := range yoloOutputLayers {
		detLayers = append(detLayers, yoloModel.Forward(l))
	}

	// Extract predictions
	yd := extractPredictions(detLayers, img.Size(), classLabels)

	fmt.Println("Detected objects:")
	for _, d := range yd {
		fmt.Println(d)
	}
	drawPredictions(img, yd)

	// Show image with predictions
	var windowTitle string
	if len(yd) > 0 {
		windowTitle = fmt.Sprintf("Detected %d objects - Press any key to close window", len(yd))
	} else {
		windowTitle = "No objects detected - Press any key to close window"
	}
	window := gocv.NewWindow(windowTitle)
	frameWidth, frameHeight := img.Size()[1], img.Size()[0]
	window.ResizeWindow(frameWidth, frameHeight)
	defer window.Close()

	window.IMShow(img)
	for {
		if window.WaitKey(1) > 0 {
			break
		}
	}
}
