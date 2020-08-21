// This example implements an OpenCV ORB algorithm to identify playing cards using go and gocv.
//
// Due to the specifics of the ORB algorithm, this method is only suitable for the face cards
// including Jacks, Queens, Kings, and Ace of Spades (in my deck), since these are feature rich and
// distinguishable, and not suitable for other cards.
//
// gocv at the moment of writing only supports default parameters for feature2d detectors
// Call: main.go [arguments]
//

package main

import (
	"fmt"
	"image"
	"image/color"
	"io/ioutil"
	"os"
	"path"
	"runtime"
	"strings"
	"time"

	"gocv.io/x/gocv"
)

const (
	usageStr = `
	Playing cards detector based on ORB algorithm. Press 'Q' to exit.
		Usage: main.go [flags]
		Flags accepted:
			-all: Detect all cards; otherwise limit detection to face cards.
	`
	detectAllFlag = "-all"
)

// Input and output parameters
const (
	camID       = 0 // Edit this for your camera
	camWidth    = 1280
	camHeight   = 720
	videoCodec  = "MJPG"
	videoFPS    = 25
	winWidth    = camWidth / 2
	winHeight   = camHeight / 2
	imgDir      = "../real_cards/train_img"
	outputVideo = "video.avi"
)

// Detection parameters
const (
	thrMatches                   = 15   // Minimum number of feature matches to detect a card
	distFactor                   = 0.75 // Magic factor
	detectInterval time.Duration = 500 * time.Millisecond
)

var detectAll bool
var faceCardPrefixes = [...]string{"Jack", "Queen", "King", "Ace of Spades"}
var defaultMask gocv.Mat = gocv.NewMat()
var (
	white = color.RGBA{255, 255, 255, 0}
	black = color.RGBA{0, 0, 0, 0}
)

type (
	// ORBPattern stores single card pattern
	ORBPattern struct {
		name  string
		img   gocv.Mat // Image
		descr gocv.Mat // ORB descriptors
	}
	// ORBPatternDetector stores a set of patterns and has an associated method
	// to match an image versus this set
	ORBPatternDetector struct {
		pats []ORBPattern
		orb  gocv.ORB
	}
)

// NewORBPatternDetector creates a new instance of ORBPatternDetector with assigned ORB
// and loads image patterns
func NewORBPatternDetector(orb gocv.ORB, dir string) ORBPatternDetector {
	pats := []ORBPattern{}

	// Set working dir to the package directory
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return ORBPatternDetector{orb: orb}
	}
	os.Chdir(path.Dir(filename))

	// Read card patterns
	items, _ := ioutil.ReadDir(imgDir)
	for _, item := range items {
		filename := item.Name()
		if !isValidName(filename) {
			continue
		}
		patImg := gocv.IMRead(imgDir+"/"+filename, gocv.IMReadGrayScale)
		if !patImg.Empty() {
			_, descr := orb.DetectAndCompute(patImg, defaultMask)
			pats = append(pats, ORBPattern{name: strings.Split(filename, ".")[0], img: patImg, descr: descr})
		}
	}

	opd := ORBPatternDetector{orb: orb, pats: pats}
	return opd
}

// Limits patterns by file name to JQK and Ace of Spades depending of arguments
func isValidName(filename string) bool {
	if detectAll {
		return true
	}
	for _, pref := range faceCardPrefixes {
		if strings.HasPrefix(filename, pref) {
			return true
		}
	}
	return false
}

// Match finds and returns a single pattern with the best match to the image, and the number of matches,
// using bruteforce matcher. Number of matches should be greater than threshold value
// Returns an empty struct and 0 in the case of no mathces detected
func (opd *ORBPatternDetector) Match(img gocv.Mat) (best ORBPattern, numMatches int) {
	if img.Empty() {
		return
	}

	// BF comparison to all patterns
	_, descr := opd.orb.DetectAndCompute(img, gocv.NewMat())
	bf := gocv.NewBFMatcher()
	bestID := -1
	for i, pat := range opd.pats {
		nMatches := numGoodMatches(bf, descr, pat.descr)
		if nMatches > numMatches && nMatches > thrMatches {
			numMatches = nMatches
			bestID = i
		}
	}

	if bestID >= 0 {
		best = opd.pats[bestID]
	}
	return
}

// Compares feature descriptions of 2 images and returns number of matches btween them
func numGoodMatches(bf gocv.BFMatcher, descr1, descr2 gocv.Mat) (num int) {
	matches := bf.KnnMatch(descr1, descr2, 2)
	for _, mtcPair := range matches {
		if mtcPair[0].Distance < distFactor*mtcPair[1].Distance {
			num++
		}
	}
	return
}

func main() {
	fmt.Println(usageStr)

	// Choose whether to detect all cards or face cards only
	if len(os.Args) >= 2 {
		detectAll = strings.ToLower(os.Args[1]) == detectAllFlag
	}

	// Start webcam first and adjust definition for better results
	webcam, _ := gocv.OpenVideoCapture(camID)
	webcam.Set(gocv.VideoCaptureFrameWidth, camWidth)
	webcam.Set(gocv.VideoCaptureFrameHeight, camHeight)
	defer webcam.Close()

	// Start video writer with the same definition as camera
	vwriter, _ := gocv.VideoWriterFile(outputVideo, videoCodec, videoFPS, camWidth, camHeight, true)
	defer vwriter.Close()

	// Initialize detector and load (card) patterns
	orb := gocv.NewORB()
	defer orb.Close()
	opd := NewORBPatternDetector(orb, imgDir)
	fmt.Println("Successfully loaded:", len(opd.pats), "patterns")

	// Output window
	window := gocv.NewWindow("ORB Detector")
	window.ResizeWindow(winWidth, winHeight)
	defer window.Close()

	img := gocv.NewMat()
	detectedClass := ""
	lastDetClass := ""
	lastDetTime := time.Now()

	for {
		webcam.Read(&img)

		img1 := img.Clone()
		pat, nMatches := opd.Match(img1)

		// Workaround for detection delay caused by video input
		if nMatches > 0 {
			detectedClass = pat.name
			lastDetClass = pat.name
			lastDetTime = time.Now()
		} else if time.Now().Sub(lastDetTime) < detectInterval {
			detectedClass = lastDetClass
		} else {
			detectedClass = ""
		}

		if detectedClass != "" {
			gocv.Rectangle(&img1, image.Rect(0, 0, 400, 40), black, -1)
			gocv.PutText(&img1, detectedClass, image.Pt(20, 30), gocv.FontHersheySimplex, 1, white, 2)
		}

		if vwriter.IsOpened() {
			vwriter.Write(img1)
		}

		window.IMShow(img1)
		if window.WaitKey(1) > 0 {
			break
		}
	}

}
