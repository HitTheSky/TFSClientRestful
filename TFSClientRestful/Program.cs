﻿using Newtonsoft.Json.Linq;
using NumSharp;
using System;
using System.Collections.Generic;
using System.Configuration;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace TFSClientRestful
{
    class Program
    {
        static void Main(string[] args)
        {
            bool DEBUG_MODE = false;
            //var testImage = new Bitmap("cat.jpg");
            //Image testImage = Image.FromFile("cat.jpg");
            //Bitmap testImage = (Bitmap)Image.FromFile("cat.jpg");             

            Bitmap testImage = (Bitmap)Image.FromFile("manypeople.jpg");
            ParseObject res = null;
            if (!DEBUG_MODE){
                res = TFSRequestRestful(testImage);
            }
            else{
                res = TestTFSRequestRestful(testImage);
            }

            if (res.NumOfDetect > 0)
            {
                Console.WriteLine("detect human: {0}", res.NumOfDetect);

            }
            else
            {
                Console.WriteLine("it could not detect human");
            }
        }

        public static ParseObject TFSRequestRestful(Bitmap bImage)
        {
            ParseObject pObj = null;
            int im_width = bImage.Width;
            int im_height = bImage.Height;

            var httpWebRequest =
                (HttpWebRequest)WebRequest.Create(ConfigurationManager.AppSettings["ServerHost2"]);
            httpWebRequest.ContentType = "application/json";
            httpWebRequest.Method = "POST";

            using (var streamWriter = new StreamWriter(httpWebRequest.GetRequestStream()))
            {
                /*NDArray bitmapBytes = GetBitmapBytes2NDArray(bImage);*/                
                string bitmapBytes = GetBitmapBytes2String(bImage);

                string json = "{\"instances\": " + bitmapBytes + "}";
                streamWriter.Write(json);
                streamWriter.Flush();
                streamWriter.Close();
            }

            var httpResponse = (HttpWebResponse)httpWebRequest.GetResponse();
            using (var streamReader = new StreamReader(httpResponse.GetResponseStream()))
            {
                var result = streamReader.ReadToEnd();
                //Console.WriteLine(result.ToString());

                JObject jobj = JObject.Parse(result);

                var dtNumDetect = jobj["predictions"][0]["num_detections"].ToObject<int>();
                var dtClassList = jobj["predictions"][0]["detection_classes"].Select(x => (int)x).ToArray();
                var dtScoreList = jobj["predictions"][0]["detection_scores"].Select(x => (float)x).ToArray();

                // if only obj is human, it will send data
                float threshold = float.Parse(ConfigurationManager.AppSettings["Threshold"]);
                pObj = new ParseObject();
                //int ppc = 0;
                for (int pid = 0; pid < dtNumDetect; pid++)
                {
                    if ((dtClassList[pid] == 1) && (threshold <= dtScoreList[pid]))
                    {
                        pObj.NumOfDetect += 1;
                        //pObj.ScoreList.SetValue(dtScoreList[pid], ppc);
                        Boxes boxes = new Boxes();
                        var dtBoxesList = jobj["predictions"][0]["detection_boxes"][pid].Select(x => (float)x).ToArray();
                        boxes.BoxX1 = dtBoxesList[0] * im_height; //p1_height
                        boxes.BoxX2 = dtBoxesList[1] * im_width; //p1_width
                        boxes.BoxY1 = dtBoxesList[2] * im_height; //p2_height
                        boxes.BoxY2 = dtBoxesList[3] * im_width; //p2_width
                        pObj.BoxList.Add(boxes);
                        //ppc++;
                    }
                }

            }

            return pObj;
        }

        public static ParseObject TestTFSRequestRestful(Bitmap bImage)
        {
            ParseObject pObj = null;
            int im_width = bImage.Width;
            int im_height = bImage.Height;

            pObj = new ParseObject();
            string res = "{\"predictions\": [{\"detection_scores\": [0.992537796, 0.0], " +
                "\"detection_classes\": [1.0, 1.0], " +
                "\"num_detections\": 1.0, " +
                "\"detection_boxes\": [[0.0457661822, 0.0729743466, 0.957826436, 0.988076806], [0.0, 0.0, 0.0, 0.0]]}]}";
            JObject jobj = JObject.Parse(res);

            var dtNumDetect = jobj["predictions"][0]["num_detections"].ToObject<int>();
            var dtClassList = jobj["predictions"][0]["detection_classes"].Select(x => (int)x).ToArray();
            var dtScoreList = jobj["predictions"][0]["detection_scores"].Select(x => (float)x).ToArray();
            float threshold = float.Parse(ConfigurationManager.AppSettings["Threshold"]);

            for (int pid = 0; pid < dtNumDetect; pid++)
            {
                if ((dtClassList[pid] == 1) && (threshold <= dtScoreList[pid]))
                {
                    pObj.NumOfDetect += 1;
                    //pObj.ScoreList.SetValue(dtScoreList[pid], ppc);
                    Boxes boxes = new Boxes();
                    var dtBoxesList = jobj["predictions"][0]["detection_boxes"][pid].Select(x => (float)x).ToArray();
                    boxes.BoxX1 = dtBoxesList[0] * im_height; //p1_height
                    boxes.BoxX2 = dtBoxesList[1] * im_width; //p1_width
                    boxes.BoxY1 = dtBoxesList[2] * im_height; //p2_height
                    boxes.BoxY2 = dtBoxesList[3] * im_width; //p2_width
                    pObj.BoxList.Add(boxes);
                    //ppc++;
                }
            }
            return pObj;
        }

        private static NDArray GetBitmapBytes2NDArray(Bitmap image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
            try
            {
                unsafe
                {
                    // Get the respective addresses
                    //IntPtr src =bmpData.Scan0;
                    byte* src = (byte*)bmpData.Scan0;
                    int sizebytes = bmpData.Stride * bmpData.Height;
                    int numBytes = bmpData.Width * bmpData.Height * 3;
                    byte[] arrayData = new byte[numBytes];
                    //System.Runtime.InteropServices.Marshal.Copy(src, byteData, 0, sizebytes);
                    int idx = 0;

                    for (int i = 0; i < sizebytes; i += 4)
                    {
                        arrayData[idx++] = src[i + 2];
                        arrayData[idx++] = src[i + 1];
                        arrayData[idx++] = src[i];
                    }
                    return NumSharp.np.array(arrayData).reshape(1, bmpData.Height, bmpData.Width, 3);
                }

            }
            finally
            {
                image.UnlockBits(bmpData);
            }
        }

        private static string GetBitmapBytes2String(Bitmap image)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, PixelFormat.Format32bppPArgb);
            try
            {
                unsafe
                {
                    // Get the respective addresses                    
                    byte* src = (byte*)bmpData.Scan0;
                    int sizebytes = bmpData.Stride * bmpData.Height;
                    int numBytes = bmpData.Width * bmpData.Height * 3;
                    byte[] arrayData = new byte[numBytes];
                    int width = bmpData.Width;
                    int height = bmpData.Height;
                    int idc = 0;
                    string nDArray = null;
                    string rowArray = null;
                    string columArray = null;

                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width * 3; w += 3)
                        {
                            string elem = String.Format("[{0}, {1}, {2}], ", src[idc + 2], src[idc + 1], src[idc]);
                            rowArray += elem;
                            idc += 4;
                        }
                        columArray += String.Format("[{0}], ", rowArray.Substring(0, rowArray.Length - 2));
                        rowArray = "";
                    }
                    nDArray = String.Format("[[{0}]]", columArray.Substring(0, columArray.Length - 2));

                    return nDArray;
                }
            }
            finally
            {
                image.UnlockBits(bmpData);
            }
        }


    }

    class Boxes
    {
        public float BoxX1 { get; set; }
        public float BoxX2 { get; set; }
        public float BoxY1 { get; set; }
        public float BoxY2 { get; set; }
    }

    class ParseObject
    {
        public ParseObject()
        {
            BoxList = new List<Boxes>();
            //ScoreList = new float[100];
        }
        public int NumOfDetect { get; set; }
        //public Array TypeOfClass { get; set; }
        //public Array ScoreList { get; set; }
        public List<Boxes> BoxList { get; set; }


    }
}
