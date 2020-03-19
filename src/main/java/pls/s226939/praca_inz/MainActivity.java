package pls.s226939.praca_inz;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar EYE_RECT_COLOR = new Scalar(255, 0, 0, 255);
    public static final int JAVA_DETECTOR = 0;

    Core.MinMaxLocResult mmG;
    Point iris;
    Rect eye_template;

    private Mat templateR;
    private Mat templateL;
    private Mat templateR_open;
    private Mat templateL_open;

    private boolean HaarEyeOpen_R = false;
    private boolean HaarEyeOpen_L = false;

    public int frame =0;
    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCORR_NORMED = 5;
    private Mat mRgba;
    private Mat mGray;


    private File mCascadeFile;
    private File cascadeFileER;
    private File cascadeFileEL;
    private File cascadeFileEyeOpen;

    private CascadeClassifier mJavaDetector;
    private CascadeClassifier mJavaDetectorEyeRight;
    private CascadeClassifier mJavaDetectorEyeLeft;
    private CascadeClassifier mJavaDetectorEyeOpen;

    private String[] mDetectorName;

    private float mRelativeFaceSize = 0.2f;
    private int faceSize = 0;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    try {

                        InputStream inputStream = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream outputStream = new FileOutputStream(mCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            outputStream.write(buffer, 0, bytesRead);
                        }
                        inputStream.close();
                        outputStream.close();

                        // ------------------ load right eye classificator -----------------------
                        InputStream iser = getResources().openRawResource(R.raw.haarcascade_righteye_2splits);
                        File cascadeDirER = getDir("cascadeER",Context.MODE_PRIVATE);
                        cascadeFileER = new File(cascadeDirER,"haarcascade_eye_right.xml");
                        FileOutputStream oser = new FileOutputStream(cascadeFileER);

                        byte[] bufferER = new byte[4096];
                        int bytesReadER;
                        while ((bytesReadER = iser.read(bufferER)) != -1) {
                            oser.write(bufferER, 0, bytesReadER);
                        }
                        iser.close();
                        oser.close();

                        // ------------------ load left eye classificator -----------------------
                        InputStream isel = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                        File cascadeDirEL = getDir("cascadeEL",Context.MODE_PRIVATE);
                        cascadeFileEL = new File(cascadeDirEL,"haarcascade_eye_left.xml");
                        FileOutputStream osel = new FileOutputStream(cascadeFileEL);

                        byte[] bufferEL = new byte[4096];
                        int bytesReadEL;
                        while ((bytesReadEL = isel.read(bufferEL)) != -1) {
                            osel.write(bufferEL, 0, bytesReadEL);
                        }
                        isel.close();
                        osel.close();

                        // ------------------ load open eye classificator -----------------------
                        InputStream opisel = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);
                        File cascadeDirEyeOpen = getDir("cascadeEyeOpen",Context.MODE_PRIVATE);
                        cascadeFileEyeOpen = new File(cascadeDirEyeOpen,"haarcascade_eye_tree_eyeglasses.xml");
                        FileOutputStream oposel = new FileOutputStream(cascadeFileEyeOpen);

                        byte[] bufferEyeOpen = new byte[4096];
                        int bytesReadEyeOpen;
                        while ((bytesReadEyeOpen = opisel.read(bufferEyeOpen)) != -1) {
                            oposel.write(bufferEyeOpen, 0, bytesReadEyeOpen);
                        }
                        opisel.close();
                        oposel.close();

                        //Face Classifier
                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier of face");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from "+ mCascadeFile.getAbsolutePath());
                        cascadeDir.delete();

                        //EyeRightClassifier
                        mJavaDetectorEyeRight = new CascadeClassifier(cascadeFileER.getAbsolutePath());
                        if (mJavaDetectorEyeRight.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier of eye right");
                            mJavaDetectorEyeRight = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileER.getAbsolutePath());
                        cascadeDirER.delete();

                        //EyeLeftClassifier
                        mJavaDetectorEyeLeft = new CascadeClassifier(cascadeFileEL.getAbsolutePath());
                        if (mJavaDetectorEyeLeft.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier of eye left");
                            mJavaDetectorEyeLeft = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileEL.getAbsolutePath());
                        cascadeDirEL.delete();

                        //EyeOpenClassifier
                        mJavaDetectorEyeOpen = new CascadeClassifier(cascadeFileEyeOpen.getAbsolutePath());
                        if (mJavaDetectorEyeOpen.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier of eye open");
                            mJavaDetectorEyeOpen = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from "+ cascadeFileEyeOpen.getAbsolutePath());
                        cascadeDirEyeOpen.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }
                    mOpenCvCameraView.setCameraIndex(1);
                    mOpenCvCameraView.enableFpsMeter();
                    mOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public MainActivity() {
        mDetectorName = new String[2];
        mDetectorName[JAVA_DETECTOR] = "Java";
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.activity_main);

        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.java_camera_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this,mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        Core.transpose(mRgba,mRgba);
        Core.flip(mRgba, mRgba,0);

        Core.flip(mRgba, mRgba,+1);

        Core.transpose(mGray,mGray);
        Core.flip(mGray,mGray,0);

        Core.flip(mGray,mGray,+1);

        if (faceSize == 0) {
            int height = mGray.rows();
            if (Math.round(height * mRelativeFaceSize) > 0) {
                faceSize = Math.round(height * mRelativeFaceSize);
            }
        }

        MatOfRect face = new MatOfRect();

        if (mJavaDetector != null)
            mJavaDetector.detectMultiScale(mGray,
                    face,
                    1.1,
                    4,
                    2,
                    new Size(faceSize, faceSize),
                    new Size());

        Rect[] facesArray = face.toArray();


        for (int i = 0; i < facesArray.length; i++) {

            Imgproc.rectangle(mRgba,
                    facesArray[i].tl(),
                    facesArray[i].br(),
                    FACE_RECT_COLOR,
                    3);

            Rect RectOfFace = facesArray[i];

            Rect eyearea_right = new Rect(
                    RectOfFace.x + RectOfFace.width / 16 ,
                    (int) (RectOfFace.y + (RectOfFace.height / 4.5)) ,
                    (RectOfFace.width - 2 * RectOfFace.width / 16) / 2,
                    (int) (RectOfFace.height / 3.0));

            Rect eyearea_left = new Rect(
                    RectOfFace.x + RectOfFace.width / 16 + ( RectOfFace.width - 2 * RectOfFace.width / 16 ) / 2 ,
                    (int) (RectOfFace.y + (RectOfFace.height / 4.5)) ,
                    (RectOfFace.width - 2 * RectOfFace.width / 16) / 2 ,
                    (int) (RectOfFace.height / 3.0));

            Imgproc.rectangle(mRgba, eyearea_left.tl(), eyearea_left.br(),
                    EYE_RECT_COLOR, 2);
            Imgproc.rectangle(mRgba, eyearea_right.tl(), eyearea_right.br(),
                    EYE_RECT_COLOR, 2);


            // int xo1 = (RectOfFace.x + RectOfFace.width / 16) + (RectOfFace.width - 2 * RectOfFace.width / 16)/2;
            // int yo1 = (int) (RectOfFace.y + (RectOfFace.height / 4.5));


            //Imgproc.putText(mRgba, "eye[" + xo1 + "," + yo1 + "]",
            //      new Point(200,  200),
            //      Core.FONT_HERSHEY_DUPLEX, 2, new Scalar(255,0, 0));

            //  Mat roi = mRgba.submat(eyearea_left);
            //Imgproc.cvtColor(roi,roi,Imgproc.COLOR_RGB2GRAY);
            // Imgproc.equalizeHist(roi,roi);
            //  MatOfPoint newcorners = new MatOfPoint();
            // Imgproc.goodFeaturesToTrack(roi, newcorners, 100, 0.6, 10);

            //  Point[] cornerpoints = newcorners.toArray();

            // for (Point points : cornerpoints) {
            //      Imgproc.circle(mRgba, points, 10, new Scalar(100,100,100), 2);
            // }
            //Imgproc.rectangle(roi, new Point(cornerpoints[0].x, cornerpoints[0].y), new Point(cornerpoints[1].x, cornerpoints[1].y), new Scalar(255, 255, 0));


            Rect rectR = get_template(mJavaDetectorEyeRight, eyearea_right);

            Rect rectL = get_template(mJavaDetectorEyeLeft, eyearea_left);


            rectR = get_template(mJavaDetectorEyeOpen, rectR, new Size(1, 1), new Size(50,50));
            templateR_open = mGray.submat(rectR);

            rectL = get_template(mJavaDetectorEyeOpen, rectL, new Size(1, 1), new Size(50,50));
            templateL_open = mGray.submat(rectL);

            if(frame<3)
            {
                templateR = get_template_pupil(mJavaDetectorEyeRight, eyearea_right);
                templateL = get_template_pupil(mJavaDetectorEyeLeft, eyearea_left);
            }
            else{
                match_eye(eyearea_right, templateR, TM_CCORR_NORMED);
                match_eye(eyearea_left, templateL, TM_CCORR_NORMED);
            }


            HaarEyeOpen_R = check_fit(templateR_open);
            HaarEyeOpen_L = check_fit(templateL_open);

            if(!HaarEyeOpen_R && !HaarEyeOpen_L){
                Imgproc.putText(mRgba, "Closed", new Point(mRgba.size().width/3, mRgba.size().height/8), Core.FONT_HERSHEY_DUPLEX, 3, new Scalar(255,0,0),5);
            }
            else if (HaarEyeOpen_R && HaarEyeOpen_L){
                Imgproc.putText(mRgba, "Open", new Point(mRgba.size().width/3, mRgba.size().height/8), Core.FONT_HERSHEY_DUPLEX, 3, new Scalar(0,255,0),5);
            }
            else if(HaarEyeOpen_L && !HaarEyeOpen_R)
            {
                Imgproc.putText(mRgba, "Right Eye Open", new Point(mRgba.size().width/5, mRgba.size().height/8), Core.FONT_HERSHEY_DUPLEX, 3, new Scalar(0,0,255),5);
            }
            else if(!HaarEyeOpen_L && HaarEyeOpen_R)
            {
                Imgproc.putText(mRgba, "Left Eye Open", new Point(mRgba.size().width/5, mRgba.size().height/8), Core.FONT_HERSHEY_DUPLEX, 3, new Scalar(0,0,255),5);
            }

        }
        frame =0;
        return mRgba;
    }

    private Rect get_template(CascadeClassifier clasificator, Rect RectAreaInterest) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(RectAreaInterest);
        MatOfRect eyes = new MatOfRect();
        iris = new Point();
        eye_template = new Rect();
        clasificator.detectMultiScale(mROI,
                eyes,
                1.15,
                4,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(10, 10),
                new Size(100,100));

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect eyeDetected = eyesArray[i];
            eyeDetected.x = RectAreaInterest.x + eyeDetected.x;
            eyeDetected.y = RectAreaInterest.y + eyeDetected.y;

            mROI = mGray.submat(eyeDetected);
            mmG = Core.minMaxLoc(mROI);

            iris.x = mmG.minLoc.x + eyeDetected.x;
            iris.y = mmG.minLoc.y + eyeDetected.y;
            eye_template = new Rect((int) iris.x -  eyeDetected.width/2, (int) iris.y -  eyeDetected.height/2,  eyeDetected.width,  eyeDetected.height);

            return eye_template;

        }
        return eye_template;
    }


    private Rect get_template(CascadeClassifier clasificator, Rect RectAreaInterest, Size min_size, Size max_size) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(RectAreaInterest);
        MatOfRect eyes = new MatOfRect();
        iris = new Point();
        eye_template = new Rect();
        clasificator.detectMultiScale(mROI,
                eyes,
                1.01,
                4,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE,
                min_size,
                max_size );

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect eyeDetected = eyesArray[i];
            eyeDetected.x = RectAreaInterest.x + eyeDetected.x;
            eyeDetected.y = RectAreaInterest.y + eyeDetected.y;

            mROI = mGray.submat(eyeDetected);
            mmG = Core.minMaxLoc(mROI);

            iris.x = mmG.minLoc.x + eyeDetected.x;
            iris.y = mmG.minLoc.y + eyeDetected.y;
            eye_template = new Rect((int) iris.x -  eyeDetected.width/2, (int) iris.y -  eyeDetected.height/2,  eyeDetected.width,  eyeDetected.height);

            return eye_template;
        }
        return eye_template;
    }

    private boolean check_fit(Mat template) {
        if (template.cols() == 0 || template.rows() == 0) {
            return false;
        }else{
            return true;
        }
    }
    private Mat get_template_pupil(CascadeClassifier clasificator, Rect area)
    {
        Mat template = new Mat();

        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();
        Rect eye_template = new Rect();

        clasificator.detectMultiScale(mROI,
                eyes,
                1.15,
                4,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT
                        | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(30, 30),
                new Size());

        Rect[] eyesArray = eyes.toArray();
        for (int i = 0; i < eyesArray.length;) {
            Rect e = eyesArray[i];
            e.x = area.x + e.x;
            e.y = area.y + e.y;
            Rect eye_only_rectangle = new Rect((int) e.tl().x,
                    (int) (e.tl().y + e.height * 0.4), (int) e.width,
                    (int) (e.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat pupil = mRgba.submat(eye_only_rectangle);


            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(pupil, mmG.minLoc, 2, new Scalar(255, 255, 255, 255), 2);

            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;


            eye_template = new Rect((int) iris.x - 24 / 2, (int) iris.y - 24 / 2, 24, 24);
            Imgproc.rectangle(mRgba, eye_template.tl(), eye_template.br(), new Scalar(255, 0, 0, 255), 2);



            if(iris.y<525)
            {
                Imgproc.putText(mRgba, "You looking UP", new Point(150,  1000),Core.FONT_HERSHEY_DUPLEX, 3.0, new Scalar(0,0, 255), 5);
            }else if(iris.y>540)
            {
                Imgproc.putText(mRgba, "You looking DOWN", new Point(150,  1000),Core.FONT_HERSHEY_DUPLEX, 3.0, new Scalar(0,0, 255), 5);

            }/*
             if(iris.x<390)
            {
                Imgproc.putText(mRgba, "You looking LEFT", new Point(150,  1000),Core.FONT_HERSHEY_DUPLEX, 3.0, new Scalar(0,0, 255), 5);

            }
            else if(iris.x>580)
            {
                Imgproc.putText(mRgba, "You looking RIGHT", new Point(150,  1000),Core.FONT_HERSHEY_DUPLEX, 3.0, new Scalar(0,0, 255), 5);

            }*/
            //Imgproc.putText(mRgba, "pupil[" + iris.x+ "," + iris.y + "]",   new Point(200,  320),  Core.FONT_HERSHEY_DUPLEX, 2.0, new Scalar(0,0, 255),2);
            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    private void match_eye(Rect area, Mat mTemplate, int type) {
        Point matchLoc;
        Mat mROI = mGray.submat(area);
        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;
        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0) {
            return ;
        }
        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        Imgproc.matchTemplate(mROI, mTemplate, mResult,Imgproc.TM_CCORR_NORMED);


        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);

        if (type == TM_SQDIFF || type == TM_SQDIFF_NORMED) {
            matchLoc = mmres.minLoc;
        } else {
            matchLoc = mmres.maxLoc;
        }

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x,
                matchLoc.y + mTemplate.rows() + area.y);

        Imgproc.rectangle(mRgba, matchLoc_tx, matchLoc_ty, new Scalar(255, 255, 0,
                255));

    }

}
