using System;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(LineRenderer))]
public class LineRendererColliderGenerator : MonoBehaviour
{
    public bool withOutline;
    private readonly List<BoxCollider> _colliders = new();
    private readonly Color _lineColor = Color.white;

    private LineRenderer _lineRenderer;
    private Outline _outline;

    public float ColliderWidth { get; set; } = -1f;
    public int ColliderIndexOffset { get; set; } = 1;

    private void Start()
    {
        _lineRenderer = GetComponent<LineRenderer>();
        if (Mathf.Approximately(ColliderWidth, -1f))
            ColliderWidth = _lineRenderer.startWidth;
        CreateColliders();

        if (withOutline)
        {
            _outline = gameObject.AddComponent<Outline>();
            _outline.OutlineMode = Outline.Mode.SilhouetteOnly;
            _outline.OutlineColor = _lineColor;
            _outline.OutlineWidth = 1f;
        }

        OnStartFinished?.Invoke();
    }

    public event Action OnStartFinished;

    private void CreateColliders()
    {
        int numPositions = _lineRenderer.positionCount;

        for (int i = 0; i < numPositions - 1; i++)
        {
            if (i % ColliderIndexOffset != 0) continue;
            
            Vector3 startPos = _lineRenderer.GetPosition(i);
            Vector3 endPos = _lineRenderer.GetPosition(i + 1);

            BoxCollider newCollider = new GameObject(_lineRenderer.name + "Collider").AddComponent<BoxCollider>();
            _colliders.Add(newCollider);

            newCollider.tag = _lineRenderer.tag;
            newCollider.gameObject.layer = _lineRenderer.gameObject.layer;

            newCollider.transform.SetParent(transform);

            Vector3 colliderPos = (transform.TransformPoint(startPos) + transform.TransformPoint(endPos)) / 2;
            newCollider.transform.position = colliderPos;

            float distance = Vector3.Distance(startPos, endPos);
            newCollider.size = new Vector3(distance, 10, ColliderWidth);

            Vector3 direction = (endPos - startPos).normalized;
            Vector3 normal = Vector3.Cross(direction, Vector3.up).normalized;
            float oldRotation = transform.rotation.eulerAngles.y;
            newCollider.transform.rotation = Quaternion.LookRotation(normal);
            newCollider.transform.Rotate(new Vector3(0, 0 - oldRotation, 0));

            newCollider.isTrigger = true;
        }
    }

    public List<BoxCollider> GetColliders()
    {
        return _colliders;
    }
}