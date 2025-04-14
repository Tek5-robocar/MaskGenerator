using System;
using System.Diagnostics;
using System.IO;
using Google.Protobuf.WellKnownTypes;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using File = UnityEngine.Windows.File;
using Random = UnityEngine.Random;

public class VisonCapture : MonoBehaviour
{
    public GameObject tracks;
    public ConfigLoader configLoader;
    public CentralLine centralLine;
    public Camera vision;
    public GameObject blackScreen;

    private float _lenghtOffset;
    private float _totalLength = 0;
    private int _posIndex = 0;
    private string _imageDirectory = "Images";
    private string _maskDirectory = "Masks";
    private string _datasetDirectory = "Dataset";
    private DateTime startTime;
    
    private void Start()
    {
        startTime = DateTime.Now;
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            centralLine.SetTrack(tracks.transform.GetChild(j).gameObject);
        }
        
        Init();
    }
    
    private void Init() 
    {
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            GameObject track = tracks.transform.GetChild(j).gameObject;
            for (int k = 0; k < track.transform.childCount; k++)
            {
                LineRenderer trackLineRenderer = track.transform.GetChild(k).GetComponent<LineRenderer>();
                if (trackLineRenderer == null) continue;
                for (int i = 0; i < trackLineRenderer.positionCount; i++)
                {
                    _totalLength += trackLineRenderer.GetPosition(i).magnitude;
                }
            }
        }
        _lenghtOffset = _totalLength / configLoader.Config.nbImage;
        Debug.Log($"Total lenght: {_totalLength}, offset: {_lenghtOffset}");
    }

    private void SetPosition(LineRenderer trackLineRenderer, int i, float localLength, float totalLength)
    {
        Vector3 A = trackLineRenderer.GetPosition(i > 0 ? i - 1 : 0);
        Vector3 B = trackLineRenderer.GetPosition(i);

        float segmentDistance = Vector3.Distance(A, B);
        float t = (localLength + trackLineRenderer.GetPosition(i).magnitude - totalLength) / segmentDistance;
        t = Mathf.Clamp01(t);

        Vector3 C = Vector3.Lerp(A, B, t);
        C.y += configLoader.Config.cameraHeight;
        C.x += Random.Range(-configLoader.Config.posZoneRadius, configLoader.Config.posZoneRadius);
        C.z += Random.Range(-configLoader.Config.posZoneRadius, configLoader.Config.posZoneRadius);
        transform.position = C;

        Vector3 forwardDirection;
    
        if (i < trackLineRenderer.positionCount - 1)
        {
            Vector3 nextPoint = trackLineRenderer.GetPosition(i + 1);
            forwardDirection = (nextPoint - B).normalized;
        }
        else
        {
            forwardDirection = (B - A).normalized;
        }

        if (forwardDirection != Vector3.zero)
        {
            Quaternion lookRotation = Quaternion.LookRotation(forwardDirection);
            Quaternion xRotation = Quaternion.Euler(configLoader.Config.cameraAngle, Random.Range(-configLoader.Config.rotationRange, configLoader.Config.rotationRange), 0f);
            transform.rotation = lookRotation * xRotation;
        }
    }

    private void CaptureCamera(bool blackScreenState, string subPath)
    {
        if (!Directory.Exists(_datasetDirectory))
        {
            Directory.CreateDirectory(_datasetDirectory);
        }

        if (!Directory.Exists(Path.Combine(_datasetDirectory, subPath)))
        {
            Directory.CreateDirectory(Path.Combine(_datasetDirectory, subPath));
        }
        
        blackScreen.SetActive(blackScreenState);
        
        RenderTexture.active = vision.targetTexture;
        vision.Render();
        Texture2D texture = new Texture2D(vision.pixelWidth, vision.pixelHeight, TextureFormat.RGB24, false);
        texture.ReadPixels(new Rect(0, 0, vision.pixelWidth, vision.pixelHeight), 0, 0);
        texture.Apply();
        
        if (!blackScreenState)
        {
            if (Random.Range(0, 100) <= configLoader.Config.blurQuantityPercent)
            {
                texture = TextureBlurrer.BlurTexture(texture, Random.Range(0, configLoader.Config.maxBlurPercent));
            }
            texture.Apply();
        }
        
        byte[] imageBytes = texture.EncodeToPNG();
        File.WriteAllBytes(Path.Combine(_datasetDirectory, subPath, $"{_posIndex}.png"), imageBytes);
    }

    private void Capture()
    {
        CaptureCamera(false, _imageDirectory);
        CaptureCamera(true, _maskDirectory);
    }

    private void Update()
    {
        if (_posIndex >= configLoader.Config.nbImage)
        {
            #if UNITY_EDITOR
            Debug.Log($"Finished in {DateTime.Now - startTime}");
            Process.Start(Path.Combine(_datasetDirectory));
            EditorApplication.isPlaying = false;
            #endif
            return;
        }
        
        float lenght = _lenghtOffset * _posIndex++;
        float localLenght = 0f;
        
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            GameObject track = tracks.transform.GetChild(j).gameObject;
            for (int k = 0; k < track.transform.childCount; k++)
            {
                track.SetActive(true);
                LineRenderer trackLineRenderer = track.transform.GetChild(k).GetComponent<LineRenderer>();
                if (trackLineRenderer == null) continue;
                for (int i = 0; i < trackLineRenderer.positionCount; i++)
                {
                    if (lenght >= localLenght && lenght <= localLenght + trackLineRenderer.GetPosition(i).magnitude)
                    {
                        SetPosition(trackLineRenderer,  i, localLenght, lenght);
                        Capture();
                        return;
                    }
                    localLenght += trackLineRenderer.GetPosition(i).magnitude;
                }
                track.SetActive(false);
            }
        }
    }
}