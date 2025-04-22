using System;
using System.Diagnostics;
using System.IO;
using UnityEditor;
using UnityEngine;
using Debug = UnityEngine.Debug;
using File = UnityEngine.Windows.File;
using Random = UnityEngine.Random;
using UnityEngine.Rendering.PostProcessing;
using MotionBlur = UnityEngine.Rendering.PostProcessing.MotionBlur;

public class VisonCapture : MonoBehaviour
{
    public GameObject tracks;
    public ConfigLoader configLoader;
    public CentralLine centralLine;
    public Camera vision;
    public GameObject blackScreen;
    public Light gameLight;
    public Shader standardShader;
    public Shader outlineFillShader;
    public PostProcessVolume postProcessVolume;
    public Material roadMaterial;

    private float _lenghtOffset;
    private float _totalLength = 0;
    private int _posIndex = 0;
    private string _imageDirectory = "Images";
    private string _maskDirectory = "Masks";
    private string _datasetDirectory = "Dataset";
    private DateTime startTime;
    private Texture2D _camTexture;
    private Rect _textureRect;
    private Outline[] _outlineList;
    private LineRenderer[] _lineList;
    
    private ColorGrading _colorGrading;
    private MotionBlur _motionBlur;
    private Grain _grain;
    
    private int densityPropertyID;

    
    private void Start()
    {
        startTime = DateTime.Now;
        for (int j = 0; j < tracks.transform.childCount; j++)
        {
            centralLine.SetTrack(tracks.transform.GetChild(j).gameObject);
        }
        
        Init();

        _camTexture = new Texture2D(vision.pixelWidth, vision.pixelHeight, TextureFormat.RGB24, false);
        _textureRect = new Rect(0, 0, vision.pixelWidth, vision.pixelHeight);

        _outlineList = tracks.GetComponentsInChildren<Outline>();
        _lineList = tracks.GetComponentsInChildren<LineRenderer>();
        
        foreach (LineRenderer lineRenderer in _lineList)
        {
            lineRenderer.widthMultiplier = configLoader.Config.lineWidthMultiplier;
        }

        densityPropertyID = Shader.PropertyToID("_MaxDensity");
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
        
        if(_colorGrading == null)
        {
            _colorGrading = postProcessVolume.profile.AddSettings<ColorGrading>();
        }
        postProcessVolume.profile.TryGetSettings(out _colorGrading);
        
        if(_grain == null)
        {
            _grain = postProcessVolume.profile.AddSettings<Grain>();
        }
        postProcessVolume.profile.TryGetSettings(out _grain);
        
        if(_motionBlur == null)
        {
            _motionBlur = postProcessVolume.profile.AddSettings<MotionBlur>();
        }
        postProcessVolume.profile.TryGetSettings(out _motionBlur);
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

    private void SetPostProcessing()
    {
        if (Random.Range(0, 100) <= configLoader.Config.colorGradientQuantityPercent)
        {
            _colorGrading.enabled.Override(true);
            _colorGrading.postExposure.Override(Random.Range(-5f, 5f));
            _colorGrading.colorFilter.Override(new Color(Random.Range(0.85f, 1.0f), Random.Range(0.85f, 1.0f),
                Random.Range(0.85f, 1.0f)));
        }
        
        if (Random.Range(0, 100) <= configLoader.Config.grainQuantityPercent)
        {
            _grain.enabled.Override(true);
            _grain.colored.Override(false);
            _grain.intensity.Override(Random.Range(0f, 1f));
            _grain.size.Override(Random.Range(0.3f, 3f));
            _grain.lumContrib.Override(Random.Range(0f, 1f));
        }
        
        if (Random.Range(0, 100) <= configLoader.Config.blurQuantityPercent)
        {
            _motionBlur.enabled.Override(true);
            _motionBlur.shutterAngle.Override(Random.Range(0f, 360f));
            _motionBlur.sampleCount.Override(Random.Range(4, 32));
        }

        SetRandomMaxDynamicShapesDensity();
    }

    private void SetRandomMaxDynamicShapesDensity()
    {
        if (Random.Range(0, 100) <= configLoader.Config.shapeQuantityPercent)
        {
            roadMaterial.SetFloat(densityPropertyID, Random.Range(0f, configLoader.Config.maxShapeDensity));
        }
        else
        {
            roadMaterial.SetFloat(densityPropertyID, 0f);
        }
    }

    private void UnsetPostProcessing()
    {
        _colorGrading.enabled.Override(false);
        _grain.enabled.Override(false);
        _motionBlur.enabled.Override(false);
    }

    private void CaptureCamera(bool blackScreenState, string subPath)
    {
        if (!Directory.Exists(_datasetDirectory))
        {
            Directory.CreateDirectory(_datasetDirectory);
        }
        
        if (!Directory.Exists(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory)))
        {
            Directory.CreateDirectory(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory));
        }

        if (!Directory.Exists(Path.Combine(_datasetDirectory,configLoader.Config.versionDirectory, subPath)))
        {
            Directory.CreateDirectory(Path.Combine(_datasetDirectory,configLoader.Config.versionDirectory, subPath));
        }
        
        blackScreen.SetActive(blackScreenState);

        if (!blackScreenState)
        {
            SetPostProcessing();
        }
        else
        {
            UnsetPostProcessing();
        }
        
        // if (!blackScreenState)
        // {
        //     foreach (Outline outline in _outlineList)
        //     {
        //         outline.outlineFillMaterial.shader = standardShader;
        //     }
        // }
        // else
        // {
        //     foreach (Outline outline in _outlineList)
        //     {
        //         outline.outlineFillMaterial.shader = outlineFillShader;
        //     }
        // }
        
        RenderTexture.active = vision.targetTexture;
        vision.Render();
        _camTexture.ReadPixels(_textureRect, 0, 0);
        _camTexture.Apply();
        
        // if (!blackScreenState)
        // {
            // if (Random.Range(0, 100) <= configLoader.Config.blurQuantityPercent)
            // {
                // _camTexture = TextureBlurrer.BlurTexture(_camTexture, Random.Range(0, configLoader.Config.maxBlurPercent));
            // }

            // _camTexture.Apply();
        // }
        
        byte[] imageBytes = _camTexture.EncodeToPNG();
        File.WriteAllBytes(Path.Combine(_datasetDirectory,configLoader.Config.versionDirectory, subPath, $"{_posIndex}.png"), imageBytes);
    }

    private void Capture()
    {
        CaptureCamera(false, _imageDirectory);
        CaptureCamera(true, _maskDirectory);
    }

    private void SetLight()
    {
        // gameLight.transform.position = new Vector3(
            // transform.position.x + Random.Range(-200f, 200f), 
            // Random.Range(5f, 50f), 
            // transform.position.z + Random.Range(-200f, 200f)
        // );
        // gameLight.transform.rotation = Quaternion.Euler(
            // Random.Range(-15f, 20f), 
            // Random.Range(0f, 360f), 
            // 0f
        // );
        // gameLight.intensity = Random.Range(0f, configLoader.Config.lightMaxIntensity);
    }

    private void Update()
    {
        if (_posIndex >= configLoader.Config.nbImage)
        {
            #if UNITY_EDITOR
            Debug.Log($"Finished in {DateTime.Now - startTime}");
            Process.Start(Path.Combine(_datasetDirectory, configLoader.Config.versionDirectory));
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
                track.SetActive(false);
            }
        }
        
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
                        // SetLight();
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