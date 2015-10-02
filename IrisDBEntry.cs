using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Emgu.CV;
using Emgu.CV.Structure;
using Emgu.Util;
using Emgu.CV.UI;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.GPU;

namespace Iris_Recognition
{
    // The entry into the db has the following fields : ID , NAME , IMAGE , IRIS CODE

    public class IrisDBEntry
    {
        public Guid id;
        public string name;
        public Image<Gray, byte> InputImage = new Image<Gray, byte>(IrisConstants.imageWidth, IrisConstants.imageHeight);
        public Image<Gray, byte> IrisCode;
    }
}