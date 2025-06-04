using System;
using System.Collections.Generic;
using UnityEngine;

public class CentralLine : MonoBehaviour
{
    private LineRenderer _centralLine;

    private int _numberCollider = 0;
    private GameObject _track;

    private int GetClosestVectorIndex(Vector3 target, Vector3[] vectors)
    {
        if (vectors == null || vectors.Length == 0)
        {
            Debug.LogError("The vector array is null or empty.");
            return -1;
        }

        int closestIndex = 0;
        float closestDistance = Vector3.Distance(target, vectors[0]);

        for (int i = 1; i < vectors.Length; i++)
        {
            float distance = Vector3.Distance(target, vectors[i]);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestIndex = i;
            }
        }

        return closestIndex;
    }

    private Vector3 RotatePointAroundPivot(Vector3 point, Vector3 pivot, Quaternion rotation)
    {
        Vector3 direction = point - pivot;
        direction = rotation * direction;
        return pivot + direction;
    }

    private void GenerateCentralLineRendererPart(GameObject part)
    {
        LineRenderer[] lines = part.GetComponentsInChildren<LineRenderer>();
        if (lines.Length != 2) return;
        List<(Vector3, Vector3)> associatedPoints = new List<(Vector3, Vector3)>();
        Vector3[] bigger = new Vector3[lines[0].positionCount > lines[1].positionCount
            ? lines[0].positionCount
            : lines[1].positionCount];
        Vector3[] smaller = new Vector3[lines[0].positionCount > lines[1].positionCount
            ? lines[1].positionCount
            : lines[0].positionCount];

        if (lines[0].positionCount > lines[1].positionCount)
        {
            lines[0].GetPositions(bigger);
            for (int i = 0; i < bigger.Length; i++)
            {
                bigger[i] += lines[0].transform.position;
                bigger[i] = RotatePointAroundPivot(bigger[i], lines[0].transform.position, lines[0].transform.rotation);
            }

            lines[1].GetPositions(smaller);
            for (int i = 0; i < smaller.Length; i++)
            {
                smaller[i] += lines[1].transform.position;
                smaller[i] =
                    RotatePointAroundPivot(smaller[i], lines[1].transform.position, lines[1].transform.rotation);
            }
        }
        else
        {
            lines[0].GetPositions(smaller);
            for (int i = 0; i < smaller.Length; i++)
            {
                smaller[i] += lines[0].transform.position;
                smaller[i] =
                    RotatePointAroundPivot(smaller[i], lines[0].transform.position, lines[0].transform.rotation);
            }

            lines[1].GetPositions(bigger);
            for (int i = 0; i < bigger.Length; i++)
            {
                bigger[i] += lines[1].transform.position;
                bigger[i] = RotatePointAroundPivot(bigger[i], lines[1].transform.position, lines[1].transform.rotation);
            }
        }

        foreach (Vector3 biggerPos in bigger)
        {
            int closestIndex = GetClosestVectorIndex(biggerPos, smaller);
            associatedPoints.Add((biggerPos, smaller[closestIndex]));
        }

        _centralLine = part.AddComponent<LineRenderer>();
        _centralLine.positionCount = associatedPoints.Count;
        _centralLine.widthMultiplier = 0f;

        for (int i = 0; i < associatedPoints.Count; i++)
        {
            Vector3 centralPoint = new Vector3(
                (associatedPoints[i].Item1.x + associatedPoints[i].Item2.x) / 2,
                (associatedPoints[i].Item1.y + associatedPoints[i].Item2.y) / 2,
                (associatedPoints[i].Item1.z + associatedPoints[i].Item2.z) / 2
            );
            _centralLine.SetPosition(i, centralPoint);
        }

        LineRendererColliderGenerator linesColliders = part.AddComponent<LineRendererColliderGenerator>();
        linesColliders.ColliderWidth = 15f;
        linesColliders.ColliderIndexOffset = 5;
        
        linesColliders.OnStartFinished += () =>
        {
            foreach (BoxCollider boxCollider in linesColliders.GetColliders())
            {
                _numberCollider++;
                boxCollider.name = $"checkpoint_{_numberCollider}";
                boxCollider.tag = "Checkpoint";
            }
        };
    }

    private void GenerateCentralLineRenderer()
    {
        for (int i = 0; i < _track.transform.childCount; i++)
        {
            GenerateCentralLineRendererPart(_track.transform.GetChild(i).gameObject);
        }
    }

    public void SetTrack(GameObject track)
    {
        _numberCollider = 0;
        this._track = track;
        GenerateCentralLineRenderer();
    }
}