using System;
using UnityEngine;
using UnityEngine.UI;

public class LinkedRawImage : MonoBehaviour
{
    public Camera myCamera;
    public RawImage linkedRawImage;

    private void FixedUpdate()
    {
        linkedRawImage.enabled = myCamera.enabled;
    }
}
