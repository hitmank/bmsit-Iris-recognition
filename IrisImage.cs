using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.GPU;
using System.Speech;
using System.Speech.Synthesis;
using System.Drawing;

namespace Iris_Recognition
{
    public class IrisImage
    {
        //for the inner pupil center
        private Point PupilCenter = new Point();

        //Outer Boundary radius
        private int OuterBoundaryRadius = 0;

        //Image path for the input image
        private String ImagePath = String.Empty;

        //The Image with the pupil coloured black
        public Image<Gray, Byte> FilledContourForSegmentation = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //Input image 
        public Image<Gray, Byte> InputImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //Clone of input image, which will be used in later operations
        public Image<Gray, Byte> inputclone;

        //To Store Smoothened image
        public Image<Gray, Byte> SmoothImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //to store the masked image --> to use contour detection
        public Image<Gray, Byte> MaskedImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //To store the detected pupil
        public Image<Gray, Byte> ContourDetectedPupilImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //the 2 color images for contour and iris boundrys-->so that we can draw color circles
        public Image<Bgr, Byte> ContourDetectedPupilImageColor = new Image<Bgr, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);
        public Image<Bgr, Byte> IrisOuterBoundaryImageColor = new Image<Bgr, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //to store the detected pupil in case contour fails
        public Image<Gray, Byte> ApproximatedPupilImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //To store the contrasted image which will be used for outer boundary detection
        public Image<Gray, Byte> IncreaseContrastImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //To store the detected iris outer boundary
        public Image<Gray, Byte> IrisOuterBoundaryImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //To store the segmented image
        public Image<Gray, Byte> SegmentedIrisImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //To store the optimised outer boundary--> so that circles come properly
        public Image<Gray, Byte> OptimisedIrisBoundaries = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);
        public Image<Gray, Byte> mask = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //used to tell whether contour detection failed or not
        public bool IsContourDetectionSatisfactory = true;

        //Creates a instance of Iris object
        public Iris Iris = new Iris();


        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////// END OF DECLARATIONS //////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        //default constructor
        public IrisImage()
        {
        }

        // Overloaded Constructor--> called when we select the image in browse. It takes the image path as the parameter
        public IrisImage(String ImagePath)
        {
            this.ImagePath = ImagePath;

            //Stores the image which we selected into InputImage
            InputImage = new Image<Gray, Byte>(ImagePath);

            //Resize the input image to 320x240 --> this is only needed if we select an image thats too large or too small
            InputImage = InputImage.Resize(IrisConstants.imageWidth, IrisConstants.imageHeight, INTER.CV_INTER_LINEAR, true);

            //Clone of input image which will be used later in processing
            inputclone = InputImage.Clone();

            //Clone the input image for smoothening
            SmoothImage = InputImage.Clone();

        }




        public void ProcessIris()
        {
            //Smooth Image----> Removes Noise
            SmoothImage = SmoothImage.SmoothGaussian(IrisConstants.SmoothConstant);

            //Mask the Pupil---> So that all the dark parts become white and other parts become black ---> so that we can use contour detection
            CvInvoke.cvInRangeS(SmoothImage, IrisConstants.LowerBoundForMask, IrisConstants.UpperBoundForMask, MaskedImage);

            //Detect Contours
            PerformContourDetection();

            //Find iris outer boundary
            DetectIrisOuterBoundary();

            //Segment iris using ceter of pupil and radius of outer iris
            SegmentIris();

            //Segmented iris is passed to Iris class where it will be used 
            Iris.LoadSegmentedIris(SegmentedIrisImage);

            //normalize iris
            Iris.NormalizeIris(PupilCenter);

            //extract the featues
            Iris.PerformFeatureExtraction();

        }



        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        ////////////////////////////////////////////////////////////////PUPIL(INNER IRIS BOUNDARY) DETECTION///////////////////////////////////////////////
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        //return true if pupil is detected on contour detection
        //false if further hough circles need to be found
        private void PerformContourDetection()
        {

            // Detected Contours will store all the contours detected in our image, Find contours will find all the contours
            // CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours
            // CV_CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments and leaves only their end points. 
            Contour<Point> detectedContours = MaskedImage.FindContours(CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, RETR_TYPE.CV_RETR_TREE);

            //Image moments help you to calculate some features like center of the object, area of the object etc---> here the object is the contour detected
            MCvMoments moments = new MCvMoments();

            //Make sure atleast one contour is detected
            while (detectedContours != null)
            {
                //Get the Moments of the Detected Contour
                moments = detectedContours.GetMoments();

                //get the area of the detected contour--> GetCentralMoment has the area
                double AreaOfDetectedContour = moments.GetCentralMoment(IrisConstants.Zero, IrisConstants.Zero);

                if (detectedContours.Total > 1)
                {
                    //((area > IrisConstants.MaxPupilArea) && (area < IrisConstants.MinPupilEyelashAreaCombined)) :
                    // to check if whole of eyelash is detected as a contour
                    //its area is greater than the pupil, but less than the pupil+eyelash area
                    //(area < IrisConstants.MinPupilArea) :
                    //to check for very small detected contours

                    if (((AreaOfDetectedContour > IrisConstants.MaxPupilArea) && (AreaOfDetectedContour < IrisConstants.MinPupilEyelashAreaCombined)) || (AreaOfDetectedContour < IrisConstants.MinPupilArea))
                    {
                        //discard the contour and process the next
                        detectedContours = detectedContours.HNext;
                        continue;
                    }

                }


                if ((AreaOfDetectedContour > IrisConstants.MinPupilArea))
                {

                    double Pupilarea = AreaOfDetectedContour;

                    //Get the Center of the Pupil --> GetSpatialMoment ---> has the center of the detected contour
                    double x = moments.GetSpatialMoment(IrisConstants.One, IrisConstants.Zero) / AreaOfDetectedContour;
                    double y = moments.GetSpatialMoment(IrisConstants.Zero, IrisConstants.One) / AreaOfDetectedContour;

                    //Store it in PupilCenter
                    PupilCenter.X = (int)x;
                    PupilCenter.Y = (int)y;

                    //Store the contour detected image in ContourDetectedPupilImage
                    ContourDetectedPupilImage = InputImage.Clone();

                    //Filled one will have the pupil coloured black
                    FilledContourForSegmentation = InputImage.Clone();


                    //--------------------------------------------------------------------
                    //Create a color image and store the grayscale contour image and convert to color, then draw colored contour on this 
                    //--------------------------------------------------------------------

                    CvInvoke.cvCvtColor(ContourDetectedPupilImage, ContourDetectedPupilImageColor, COLOR_CONVERSION.GRAY2BGR);

                    //Draw the contour over the pupil 
                    // ContourDetectedPupilImage.Draw(detectedContours, new Gray(255), IrisConstants.Zero);

                    //Fill the center of the pupil black--> -1 indicates fill
                    FilledContourForSegmentation.Draw(detectedContours, new Gray(IrisConstants.Zero), -1);

                    //DRAW the Colored circle in red
                    ContourDetectedPupilImageColor.Draw(detectedContours, new Bgr(0, 0, 255), 2);


                    //If the eyebrow is detected then apply hough transform

                    if (AreaOfDetectedContour > IrisConstants.MinPupilEyelashAreaCombined)
                    {
                        //Draw the contour white
                        ContourDetectedPupilImageColor.Draw(detectedContours, new Bgr(255, 255, 255), 2);

                        //make the flag false
                        IsContourDetectionSatisfactory = false;

                        //Clone the image to apply hough transform
                        ApproximatedPupilImage = ContourDetectedPupilImage.Clone();

                        //Create image to store the approximated pupil
                        Image<Gray, Byte> ApproximatedPupilImageWithContrast = ApproximatedPupilImage.Clone();

                        //Contrast the image for histogram
                        ApproximatedPupilImageWithContrast._EqualizeHist();

                        //Perform Hough Trasform
                        PerformHoughTransform(ApproximatedPupilImageWithContrast,
                            IrisConstants.HoughCircleThreshold, IrisConstants.MinPupilHoughCircleAccumulator, IrisConstants.MaxPupilHoughCircleAccumulator,
                            IrisConstants.PupilHoughCircleResolution, IrisConstants.MinPupilHoughCircleDistance,
                            IrisConstants.MinPupilHoughCircleRadius, IrisConstants.MaxPupilHoughCircleRadius, HoughTransformFlag.Pupil);
                    }
                    break;
                }
                detectedContours = detectedContours.HNext;
            }
        }




        private void DetectIrisOuterBoundary()
        {
            IncreaseContrastImage = InputImage.Clone();
            IncreaseContrastImage._EqualizeHist();
            Image<Gray, Byte> ContrastImageForHoughCircle = IncreaseContrastImage.Clone();
            PerformHoughTransform(ContrastImageForHoughCircle,
                IrisConstants.HoughCircleThreshold, IrisConstants.MinOuterBoundaryHoughCircleAccumulator, IrisConstants.MaxOuterBoundaryHoughCircleAccumulator,
                IrisConstants.OuterBoundaryHoughCircleResolution, IrisConstants.MinOuterBoundaryHoughCircleDistance,
                IrisConstants.MinOuterBoundaryHoughCircleRadius, IrisConstants.MaxOuterBoundaryHoughCircleRadius, HoughTransformFlag.IrisOuterBoundary);

        }


        // Hough transform

        private void PerformHoughTransform(Image<Gray, Byte> ImageToProcess,
            int HoughCircleThreshold, int MinAccumulator, int MaxAccumulator,
            double Resolution, double MinDistance,
            int MinRadius, int MaxRadius, HoughTransformFlag Mode)
        {
            //Accumulator value
            int currentAccumulator = MinAccumulator;
            //Threshold value
            Gray threshold = new Gray(HoughCircleThreshold);


            //start incrementing the accumilator till we find a proper circle
            while (currentAccumulator < MaxAccumulator)
            {
                Gray accumulator = new Gray(currentAccumulator);

                //apply hough circle
                CircleF[] detectedCircles = ImageToProcess.HoughCircles(threshold, accumulator, Resolution, MinDistance, MinRadius, MaxRadius)[0];
                if (detectedCircles.Length == 1)
                {
                    foreach (CircleF circle in detectedCircles)
                    {
                        switch (Mode)
                        {
                            //Hough Transform for pupil
                            case HoughTransformFlag.Pupil:
                                ImageToProcess.Draw(circle, new Gray(200), 2);
                                ContourDetectedPupilImageColor.Draw(circle, new Bgr(0, 0, 255), 2);
                                PupilCenter.X = (int)circle.Center.X;
                                PupilCenter.Y = (int)circle.Center.Y;
                                break;

                            //For outer Boundary
                            case HoughTransformFlag.IrisOuterBoundary:
                                IrisOuterBoundaryImageColor = ContourDetectedPupilImageColor.Clone();
                                ImageToProcess.Draw(circle, new Gray(200), 2);

                                //Radius of the outer boundary
                                OuterBoundaryRadius = (int)circle.Radius;

                                CvInvoke.cvCircle(IrisOuterBoundaryImageColor, PupilCenter, OuterBoundaryRadius, IrisConstants.WhiteColor, 2, Emgu.CV.CvEnum.LINE_TYPE.CV_AA, 0);
                                break;
                        }
                    }

                    break;
                }

                else if (detectedCircles.Length != 1)
                {
                    currentAccumulator++;
                }

            }
        }

        private void SegmentIris()
        {
            //Clone the filled contour
            Image<Gray, Byte> InputImageCloneOne = FilledContourForSegmentation.Clone();

            Image<Gray, Byte> InputImageCloneTwo = FilledContourForSegmentation.Clone();
            MCvScalar k = new MCvScalar(255, 255, 255);

            //Draw the circle for mask in white
            CvInvoke.cvCircle(mask, PupilCenter, OuterBoundaryRadius, IrisConstants.WhiteColor, -1, Emgu.CV.CvEnum.LINE_TYPE.CV_AA, 0);

            //Create the optimised circle using pupil center and outer boundary iris -> so that circles appear proper around the iris
            if (IsContourDetectionSatisfactory)
            {
                OptimisedIrisBoundaries = FilledContourForSegmentation.Clone();
                CvInvoke.cvCircle(OptimisedIrisBoundaries, PupilCenter, OuterBoundaryRadius, IrisConstants.WhiteColor, 2, Emgu.CV.CvEnum.LINE_TYPE.CV_AA, 0);
            }
            else
            {
                OptimisedIrisBoundaries = ApproximatedPupilImage.Clone();
                CvInvoke.cvCircle(OptimisedIrisBoundaries, PupilCenter, OuterBoundaryRadius, IrisConstants.WhiteColor, 2, Emgu.CV.CvEnum.LINE_TYPE.CV_AA, 0);
            }

            //now make the mask circle black
            CvInvoke.cvNot(mask, mask);

            //Subtract the input image and filled contour image over the mask created
            CvInvoke.cvSub(InputImage, InputImageCloneOne, InputImageCloneTwo, mask);

            //Put clonetwo to segmented image
            CvInvoke.cvCopy(InputImageCloneTwo, SegmentedIrisImage, new IntPtr(0));

        }

        //scope
        public void Load(String ImagePath)
        {
            this.ImagePath = ImagePath;
            InputImage = new Image<Gray, Byte>(ImagePath);
            InputImage = InputImage.Resize(IrisConstants.imageWidth, IrisConstants.imageHeight, INTER.CV_INTER_LINEAR, true);
            inputclone = InputImage.Clone();
            SmoothImage = InputImage.Clone();
            FilledContourForSegmentation = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);
        }
    }
}
