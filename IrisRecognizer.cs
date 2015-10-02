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
using Iris_Recognition;
using System.Threading;

namespace Iris_Recognition
{
    public class IrisRecognizer
    {
        private IrisDB DB = new IrisDB();
        private Thread Iristhread2;
        private Thread Iristhread1;
        public double hammingDistance;
        private int Xorsum;
        public bool enroll(IrisImage irisImage, string name)
        {
            try
            {
                irisImage.ProcessIris();
            }
            catch(Exception e)
            {
                //throw new Exception("Processing Error. Iris Rejected.");
            }
            try
            {                
                IrisDBEntry Entry = new IrisDBEntry();
                Entry.id = Guid.NewGuid();
                Entry.name = name;
                Entry.InputImage = irisImage.InputImage;
                Entry.IrisCode = irisImage.Iris.FullIrisCode;
                DB.Insert(Entry);
                return true;
            }
            catch (Exception e)
            {
                throw new Exception("Database Error!");
            }
        }

        private void ThreadProcess(IrisImage iris)
        {
            iris.ProcessIris();
            Thread.CurrentThread.Abort();
        }

       


        public bool Match(Image<Gray,Byte> IrisCode1,Image<Gray,Byte> IrisCode2)
        {
            Image<Gray, Byte> XORImage = new Image<Gray, Byte>(360, IrisConstants.imageHeight);
            XORImage.SetValue(0.0);
            CvInvoke.cvXor(IrisCode1,IrisCode2, XORImage, IntPtr.Zero);
           Xorsum=0;
            for(int w=0;w<360;w++)
            {
                for(int h=0;h<240;h++)
                {
                    if(XORImage.Data[h,w,0]==255)
                    {
                        Xorsum++;
                    }
                }
            }

            hammingDistance=(double)Xorsum/86400;

            if (hammingDistance > 0.29)
                return false;
            else
                return true;
        }

        public bool Match(IrisImage FirstImage, IrisImage SecondImage)
        {
            Iristhread1 = new Thread(() => ThreadProcess(FirstImage));
            Iristhread1.Start();
            Iristhread2 = new Thread(() => ThreadProcess(SecondImage));
            Iristhread2.Start();

            while(Iristhread1.ThreadState != ThreadState.Stopped || Iristhread2.ThreadState != ThreadState.Stopped)
            {
                Thread.Sleep(10);
            }
            //Image<Gray, Byte> andImage = new Image<Gray, Byte>(60, 240);
            //CvInvoke.cvAnd(mask1, mask2, andImage, IntPtr.Zero);
            return Match(FirstImage.Iris.FullIrisCode, SecondImage.Iris.FullIrisCode);
        }

        private bool MatchThread(Image<Gray,Byte> IrisCode1,Image<Gray,Byte> IrisCode2)
        {
            return true;
        }

        public IrisDBEntry Match(IrisImage FirstImage)
        {            
            FirstImage.ProcessIris();
            int count = DB.Count();
            List<IrisDBEntry> Entries;
            for (int i = 0; i < count; i += Math.Min(6, count - i))
            {
                Entries = DB.Select();
                foreach (var entry in Entries)
                {
                    bool result;
                    //Thread MatchThread = new Thread(() => { result = MatchThread(FirstImage.Iris.FullIrisCode, entry.IrisCode); });
                    result = Match(FirstImage.Iris.FullIrisCode, entry.IrisCode);
                    if (result)
                    {
                        return entry;
                    }
                }
            }
            return null;
        }
    }
}
