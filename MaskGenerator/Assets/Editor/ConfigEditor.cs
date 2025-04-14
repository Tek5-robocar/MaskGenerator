using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;

internal struct SliderElement
{
    public string label;
    public float value;
    public float minValue;
    public float maxValue;
}

public class ConfigEditor : EditorWindow
{
    public bool save;

    public List<Texture> textureList;

    // public int nbLoop = 1;
    public float timeOffset = 1f;

    // public int imageWidth = 1920;
    // public int imageHeight = 1080;
    // public float imageToMatrixScale = 1f;
    public float additionalRotationFieldOfVision = 90;
    public int numberAdditionalRotation = 1;
    public int visionDepth = 150;
    public Vector3 cameraRotation;
    public string version = CreateLatestVersion();

    private readonly List<SliderElement> _sliders = new();


    private void OnEnable()
    {
        _sliders.Add(new SliderElement { label = "nbLoop", value = 1f, minValue = 0f, maxValue = 10f });
        _sliders.Add(new SliderElement { label = "imageWidth", value = 1920f, minValue = 50f, maxValue = 1920f });
        _sliders.Add(new SliderElement { label = "imageHeight", value = 1080f, minValue = 50f, maxValue = 1080f });
        _sliders.Add(new SliderElement { label = "imageToMatrixScale", value = 1f, minValue = -2f, maxValue = 2f });
    }

    private void OnGUI()
    {
        VersionHandler();
        DrawSeparator();
        DataAugmentationHandler();
        DrawSeparator();
        SettingsHandler();
        // lines.Append($"DATA_AUGMENTATION");
        // lines.Append($"texture:{string.Join(';', textureList.Select(texture => texture.name))}");
        // lines.Append($"additionalRotationFieldOfVision:{additionalRotationFieldOfVision}");
        // lines.Append($"numberAdditionalRotation:{numberAdditionalRotation}");
        //
        // lines.Append($"SETTINGS");
        // lines.Append($"loop:{nbLoop}");
        // lines.Append($"timeOffset:{timeOffset}");
        // lines.Append($"imageWidth:{imageWidth}");
        // lines.Append($"imageHeight:{imageHeight}");
        // lines.Append($"imageToMatrixScale:{imageToMatrixScale}");
        // lines.Append($"visionDepth:{visionDepth}");
        // lines.Append($"cameraTransform:{_cameraTransform}");
    }

    [MenuItem("Window/ConfigEditor")]
    public static void ShowWindow()
    {
        GetWindow(typeof(ConfigEditor));
    }

    private void DrawSeparator()
    {
        GUILayout.Space(10);
        var rect = EditorGUILayout.GetControlRect(false, 2);
        EditorGUI.DrawRect(rect, Color.gray);
        GUILayout.Space(10);
    }

    private bool AreAllTextureSelected()
    {
        foreach (var texture in textureList)
            if (!texture)
                return false;
        return true;
    }

    private void TextureHandler()
    {
        GUILayout.Label("Load Multiple Textures");
        for (var i = 0; i < textureList.Count; i++)
        {
            EditorGUILayout.BeginHorizontal();

            textureList[i] =
                (Texture2D)EditorGUILayout.ObjectField("Texture " + (i + 1), textureList[i], typeof(Texture2D), false);

            if (GUILayout.Button("Remove", GUILayout.Width(60))) textureList.RemoveAt(i);

            EditorGUILayout.EndHorizontal();
        }

        if (GUILayout.Button("Add Texture") && AreAllTextureSelected()) textureList.Add(null);

        if (GUILayout.Button("Clear All Textures")) textureList.Clear();
    }

    private void DisplaySlider(string label)
    {
        GUILayout.BeginHorizontal();
        EditorGUILayout.LabelField(label);
        GUILayout.Space(10);
        var value = _sliders.Find(x => x.label == label);
        value.value = EditorGUILayout.Slider(value.value, value.minValue, value.maxValue);
        GUILayout.EndHorizontal();
    }

    private void DataAugmentationHandler()
    {
        GUILayout.Label("DATA_AUGMENTATION", EditorStyles.boldLabel);
        GUILayout.BeginHorizontal();
        additionalRotationFieldOfVision = EditorGUILayout.Slider("additionalRotationFieldOfVision",
            additionalRotationFieldOfVision, 0, 180, GUILayout.Width(100));
        GUILayout.EndHorizontal();
        numberAdditionalRotation =
            EditorGUILayout.IntSlider("numberAdditionalRotation", numberAdditionalRotation, 0, 180);
        TextureHandler();
    }

    private void SettingsHandler()
    {
        GUILayout.Label("SETTINGS", EditorStyles.boldLabel);
        DisplaySlider("nbLoop");
        DisplaySlider("imageToMatrixScale");
        DisplaySlider("imageWidth");
        DisplaySlider("imageHeight");
        // nbLoop = EditorGUILayout.IntSlider("nbLoop", nbLoop, 1, 10);
        timeOffset = EditorGUILayout.Slider("timeOffset", timeOffset, 0f, 5f);
        // imageToMatrixScale = EditorGUILayout.Slider("imageToMatrixScale", imageToMatrixScale, 0f, 1f);
        // imageWidth = EditorGUILayout.IntSlider("imageWidth", imageWidth, 0, 180);
        // imageHeight = EditorGUILayout.IntSlider("imageHeight", imageHeight, 0, 180);
        visionDepth = EditorGUILayout.IntSlider("visionDepth", visionDepth, 0, 180);
        cameraRotation = EditorGUILayout.Vector3Field("cameraRotation", cameraRotation);
    }

    private static string CreateLatestVersion()
    {
        if (!Directory.Exists("Datasets"))
            return "0.0.0";

        var directories = Directory.GetDirectories("Datasets");
        string[] version = { "0", "0", "0" };

        foreach (var directory in directories)
        {
            int[] dirVersion = { 0, 0, 0 };
            var i = 0;
            foreach (var se in directory.Split('\\')[^1].Split('.')) dirVersion[i++] = int.Parse(se);
            if (dirVersion[0] > int.Parse(version[0]))
            {
                version[0] = dirVersion[0].ToString();
                version[1] = dirVersion[1].ToString();
                version[2] = (dirVersion[2] + 1).ToString();
            }
            else if (dirVersion[0] == int.Parse(version[0]) && dirVersion[1] > int.Parse(version[1]))
            {
                version[1] = dirVersion[1].ToString();
                version[2] = (dirVersion[2] + 1).ToString();
            }
            else if (dirVersion[0] == int.Parse(version[0]) && dirVersion[0] == int.Parse(version[1]) &&
                     dirVersion[2] >= int.Parse(version[2]))
            {
                version[2] = (dirVersion[2] + 1).ToString();
            }
        }

        return string.Join(".", version);
    }

    private void VersionHandler()
    {
        GUILayout.Label("VERSION", EditorStyles.boldLabel);
        version = EditorGUILayout.TextField("Version", version);
    }
}