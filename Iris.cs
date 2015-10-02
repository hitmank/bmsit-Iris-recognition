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
    //------------------------------------------------------------------------------------------------------------------------------------------------------
    // This Class has the functions for Normalisation, feature extraction , creating the full iris and generating the binary number iris code for 1st filter
    //------------------------------------------------------------------------------------------------------------------------------------------------------

    public class Iris
    {

        public String IrisCodeOfFilter1, text;

        //The mask of the normalsied image
        public Image<Gray, Byte> normalizationMask = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //the normalised image got after segmentation ( un-cut)
        public Image<Gray, Byte> FullNormalImage = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);

        //The Cut normalised iris
        public Image<Gray, Byte> NormalisedIris = new Image<Gray, byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);

        //The segmented iris -> passed from IrisImage class
        public Image<Gray, Byte> InputSegmentedIris = new Image<Gray, Byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);


        //Images to store the 6 filter outputs
        public Image<Gray, Byte> GaborFilter1 = new Image<Gray, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);
        public Image<Gray, Byte> GaborFilter2 = new Image<Gray, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);
        public Image<Gray, Byte> GaborFilter3 = new Image<Gray, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);
        public Image<Gray, Byte> GaborFilter4 = new Image<Gray, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);
        public Image<Gray, Byte> GaborFilter5 = new Image<Gray, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);
        public Image<Gray, Byte> GaborFilter6 = new Image<Gray, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);


        //List where we store the above 6 filters
        public List<Image<Gray, Byte>> ExtractedFeatures = new List<Image<Gray, Byte>>(IrisConstants.NumGaborFilters);


        //The image which stores the combined iriscode
        public Image<Gray, Byte> FullIrisCode = new Image<Gray, Byte>(360, 240);

        //The leftMost Point which will used to cut the normalised image
        public int LeftMostPoint;


        //Constructors
        public Iris()
        {
        }


        public Iris(Image<Gray, Byte> InputSegmentedIris)
        {
            this.InputSegmentedIris = InputSegmentedIris.Clone();
        }


        public void LoadSegmentedIris(Image<Gray, Byte> InputSegmentedIris)
        {
            this.InputSegmentedIris = InputSegmentedIris.Clone();
        }


        /// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// //////////////////////////////////////////////////////////////////Normalisation Function///////////////////////////////////////////////////////////////////////////////////
        /// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        public void NormalizeIris(Point CenterOfPupil)
        {
            //Normalisation done to segmented iris-Full normalised iris
            CvInvoke.cvLogPolar(InputSegmentedIris, FullNormalImage, CenterOfPupil, IrisConstants.ScaleMagnitude, IrisConstants.Interpolation + IrisConstants.Warp);


            //////////////////// CUTTING THE NORMALISED IMAGE /////////////////////////////////
            // this range is to search for black part of image 
            MCvScalar le = new MCvScalar(0, 0, 0);
            MCvScalar ue = new MCvScalar(0, 0, 0);

            //masks the normalsied image
            CvInvoke.cvInRangeS(FullNormalImage, le, ue, normalizationMask);

            // Convert to color so that we can find the leftmost point
            Image<Bgr, Byte> NormalColor = new Image<Bgr, Byte>(320, 240);

            CvInvoke.cvCvtColor(normalizationMask, NormalColor, COLOR_CONVERSION.GRAY2BGR);
            Bitmap Gabor = NormalColor.ToBitmap();
            LeftMostPoint = IrisConstants.imageWidth;

            for (int a = 0; a < 240; a++)
            {
                for (int b = 100; b < 320; b++)
                {
                    //if the pixel is black the see if that point is less than the LeftMostPoint
                    if (!(Gabor.GetPixel(b, a).A.ToString() == "255" && Gabor.GetPixel(b, a).B.ToString() == "255" &&
                                      Gabor.GetPixel(b, a).G.ToString() == "255" && Gabor.GetPixel(b, a).R.ToString() == "255"))
                    {


                        if (b < LeftMostPoint)
                        {
                            LeftMostPoint = b;
                        }
                        break;
                    }
                }
            }


            Image<Gray, Byte> NormalClone = FullNormalImage.Clone();
            //Set ROI of the full normalimage based on the leftmost point and size of the strip ( 60x240)
            CvInvoke.cvSetImageROI(NormalClone, new Rectangle(new Point(LeftMostPoint, 0), new System.Drawing.Size(IrisConstants.CutImageWidth, 240)));

            //copy that ROI into Normalised iris
            CvInvoke.cvCopy(NormalClone, NormalisedIris, IntPtr.Zero);

        }


        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /// //////////////////////////////////////////////////////////////////FEATURE EXTRACTION///////////////////////////////////////////////////////////////////////////////////
        /// ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


        public void PerformFeatureExtraction()
        {

            //APPLY THE GABOR FILTERS
            CvInvoke.cvFilter2D(NormalisedIris, GaborFilter1, IrisConstants.filter1r, IrisConstants.GaborFilterAnchor);
            CvInvoke.cvFilter2D(NormalisedIris, GaborFilter2, IrisConstants.filter1i, IrisConstants.GaborFilterAnchor);
            CvInvoke.cvFilter2D(NormalisedIris, GaborFilter3, IrisConstants.filter2r, IrisConstants.GaborFilterAnchor);
            CvInvoke.cvFilter2D(NormalisedIris, GaborFilter4, IrisConstants.filter2i, IrisConstants.GaborFilterAnchor);
            CvInvoke.cvFilter2D(NormalisedIris, GaborFilter5, IrisConstants.filter3r, IrisConstants.GaborFilterAnchor);
            CvInvoke.cvFilter2D(NormalisedIris, GaborFilter6, IrisConstants.filter3i, IrisConstants.GaborFilterAnchor);

            //APPLY THRESHOLD
            CvInvoke.cvThreshold(GaborFilter1, GaborFilter1, 0, 255, THRESH.CV_THRESH_BINARY);
            CvInvoke.cvThreshold(GaborFilter2, GaborFilter2, 0, 255, THRESH.CV_THRESH_BINARY);
            CvInvoke.cvThreshold(GaborFilter3, GaborFilter3, 0, 255, THRESH.CV_THRESH_BINARY);
            CvInvoke.cvThreshold(GaborFilter4, GaborFilter4, 0, 255, THRESH.CV_THRESH_BINARY);
            CvInvoke.cvThreshold(GaborFilter5, GaborFilter5, 0, 255, THRESH.CV_THRESH_BINARY);
            CvInvoke.cvThreshold(GaborFilter6, GaborFilter6, 0, 255, THRESH.CV_THRESH_BINARY);


            //generate iris code of first filter
            IrisCodeGenerator(GaborFilter1, 1);

            // Add the filtered images to the list          
            ExtractedFeatures.Add(GaborFilter1);
            ExtractedFeatures.Add(GaborFilter2);
            ExtractedFeatures.Add(GaborFilter3);
            ExtractedFeatures.Add(GaborFilter4);
            ExtractedFeatures.Add(GaborFilter5);
            ExtractedFeatures.Add(GaborFilter6);


            //FULL IRIS CODE GENERATION
            // here we are basically setting roi for each filtered image and copyinginto FullIrisCode

            CvInvoke.cvSetImageROI(FullIrisCode, new Rectangle(new Point(0, 0), new System.Drawing.Size(60, 240)));
            CvInvoke.cvCopy(GaborFilter1, FullIrisCode, new IntPtr(0));
            CvInvoke.cvResetImageROI(FullIrisCode);
            CvInvoke.cvSetImageROI(FullIrisCode, new Rectangle(new Point(60, 0), new System.Drawing.Size(60, 240)));
            CvInvoke.cvCopy(GaborFilter2, FullIrisCode, new IntPtr(0));
            CvInvoke.cvResetImageROI(FullIrisCode);
            CvInvoke.cvSetImageROI(FullIrisCode, new Rectangle(new Point(120, 0), new System.Drawing.Size(60, 240)));
            CvInvoke.cvCopy(GaborFilter3, FullIrisCode, new IntPtr(0));
            CvInvoke.cvResetImageROI(FullIrisCode);
            CvInvoke.cvSetImageROI(FullIrisCode, new Rectangle(new Point(180, 0), new System.Drawing.Size(60, 240)));
            CvInvoke.cvCopy(GaborFilter4, FullIrisCode, new IntPtr(0));
            CvInvoke.cvResetImageROI(FullIrisCode);
            CvInvoke.cvSetImageROI(FullIrisCode, new Rectangle(new Point(240, 0), new System.Drawing.Size(60, 240)));
            CvInvoke.cvCopy(GaborFilter5, FullIrisCode, new IntPtr(0));
            CvInvoke.cvResetImageROI(FullIrisCode);
            CvInvoke.cvSetImageROI(FullIrisCode, new Rectangle(new Point(300, 0), new System.Drawing.Size(60, 240)));
            CvInvoke.cvCopy(GaborFilter6, FullIrisCode, new IntPtr(0));
            CvInvoke.cvResetImageROI(FullIrisCode);

        }


        // Generate Iris code ( BINARY NUMBER ) for the first image
        // GO pixel by pixel and see the color, if black make it 0, else make it 1
        public void IrisCodeGenerator(Image<Gray, Byte> CutGaborImage, int filterNumber)
        {
            Image<Bgr, Byte> tempImage = new Image<Bgr, Byte>(IrisConstants.CutImageWidth, IrisConstants.imageHeight);
            CvInvoke.cvCvtColor(CutGaborImage, tempImage, COLOR_CONVERSION.GRAY2BGR);
            Bitmap BitmapofGabor = tempImage.ToBitmap();

            for (int p = 0; p < BitmapofGabor.Width; p++)
            {

                for (int q = 0; q < BitmapofGabor.Height; q++)
                {

                    if (BitmapofGabor.GetPixel(p, q).A.ToString() == "255" &&

                         BitmapofGabor.GetPixel(p, q).B.ToString() == "255" &&

                         BitmapofGabor.GetPixel(p, q).G.ToString() == "255" &&

                         BitmapofGabor.GetPixel(p, q).R.ToString() == "255")
                    {
                        text = text + "0";
                    }

                    else
                    {
                        text = text + "1";
                    }
                }
            }
            IrisCodeOfFilter1 = text;
        }

    }

}
