"use client"

import type React from "react"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Upload, X } from "lucide-react"
import Image from "next/image"
// Add import for the DiagnosisBadge component
import { DiagnosisBadge } from "./components/diagnosis-badge"
// Import the StatsSummary component
import { StatsSummary } from "./components/stats-summary"

// Update the Prediction interface to match your API response format
interface Prediction {
  label: string
  confidence: number
}

// Update the ImagePrediction interface to match your API response format
interface ImagePrediction {
  id: string
  file: File
  preview: string
  prediction?: Prediction
  isLoading: boolean
  error?: string
}

export default function ImageClassifier() {
  const [images, setImages] = useState<ImagePrediction[]>([])
  // Add a state to track all predictions
  const [allPredictions, setAllPredictions] = useState<Prediction[]>([])
  const [useMockApi, setUseMockApi] = useState(false)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const newFiles = Array.from(e.target.files).map((file) => ({
        id: Math.random().toString(36).substring(2, 9),
        file,
        preview: URL.createObjectURL(file),
        isLoading: false,
      }))
      setImages((prev) => [...prev, ...newFiles])
    }
  }

  const removeImage = (id: string) => {
    setImages((prev) => {
      const filtered = prev.filter((img) => img.id !== id)
      return filtered
    })
  }

  // Update the getPredictions function to use either the real or mock API
  const getPredictions = async () => {
    // Update all images to loading state
    setImages((prev) =>
      prev.map((img) => ({
        ...img,
        isLoading: true,
        error: undefined,
      })),
    )

    try {
      // Create a FormData object with all images
      const formData = new FormData()

      // Make sure we're using the correct field name expected by the API
      images.forEach((img) => {
        formData.append(`file`, img.file)
      })

      // Choose which endpoint to use based on the toggle
      const endpoint = useMockApi ? "/api/mock" : "/api/proxy"
      console.log(`Sending request to ${endpoint} with files:`, images.length)

      const response = await fetch(endpoint, {
        method: "POST",
        body: formData,
      })

      const responseData = await response.json()

      // Check if the response contains an error
      if (response.status !== 200 || responseData.error) {
        console.error("API error:", responseData)
        throw new Error(responseData.error || "Failed to get predictions")
      }

      // Update images with predictions
      if (responseData.predictions && Array.isArray(responseData.predictions)) {
        setAllPredictions(responseData.predictions)
        setImages((prev) =>
          prev.map((img, index) => ({
            ...img,
            prediction: index < responseData.predictions.length ? responseData.predictions[index] : undefined,
            isLoading: false,
          })),
        )
      } else {
        throw new Error("API response didn't contain predictions array")
      }
    } catch (error) {
      console.error("Error fetching predictions:", error)

      // Handle errors
      setImages((prev) =>
        prev.map((img) => ({
          ...img,
          error: error instanceof Error ? error.message : "Unknown error",
          isLoading: false,
        })),
      )
    }
  }

  return (
    <div className="container mx-auto py-8 px-4">
      {/* Update the title to be more specific to the medical application */}
      <h1 className="text-3xl font-bold mb-6 text-center">Chest X-Ray Classification</h1>

      {/* Add a description below the title */}
      <p className="text-center text-gray-600 mb-8">
        Upload chest X-ray images to detect Covid, Viral Pneumonia, or Normal conditions
      </p>

      {/* Add this right before the drag and drop area */}
      <div className="flex justify-center mb-4">
        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id="use-mock"
            checked={useMockApi}
            onChange={(e) => setUseMockApi(e.target.checked)}
            className="h-4 w-4 rounded border-gray-300 text-primary focus:ring-primary"
          />
          <label htmlFor="use-mock" className="text-sm text-gray-600">
            Use sample data (check this if the real API is unavailable)
          </label>
        </div>
      </div>

      <div className="mb-8">
        <div className="flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-lg p-12 text-center">
          <Upload className="h-12 w-12 text-gray-400 mb-4" />
          <p className="text-lg mb-4">Drag and drop images here or click to browse</p>
          <Button onClick={() => document.getElementById("file-upload")?.click()} variant="outline" className="mb-2">
            Select Images
          </Button>
          <input
            id="file-upload"
            type="file"
            multiple
            accept="image/*"
            className="hidden"
            onChange={handleFileChange}
          />
          <p className="text-sm text-gray-500">Supports multiple images</p>
        </div>
      </div>

      {images.length > 0 && (
        <>
          <div className="flex justify-center mb-8">
            <Button onClick={getPredictions} disabled={images.length === 0}>
              Get Predictions
            </Button>
          </div>
          {allPredictions.length > 0 && <StatsSummary predictions={allPredictions} />}

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {images.map((img) => (
              <Card key={img.id} className="overflow-hidden">
                <div className="relative aspect-square">
                  <Image src={img.preview || "/placeholder.svg"} alt="Preview" fill className="object-cover" />
                  <button
                    onClick={() => removeImage(img.id)}
                    className="absolute top-2 right-2 bg-black bg-opacity-50 rounded-full p-1 text-white"
                  >
                    <X className="h-5 w-5" />
                  </button>
                </div>
                <CardContent className="p-3">
                  {img.isLoading ? (
                    <div className="flex justify-center py-4">
                      <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-primary"></div>
                    </div>
                  ) : img.error ? (
                    <div className="text-red-500 text-center py-2">{img.error}</div>
                  ) : img.prediction ? (
                    // Add the DiagnosisBadge to the card content
                    <div className="space-y-2">
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Diagnosis:</span>
                        <DiagnosisBadge label={img.prediction.label} />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="font-medium">Confidence:</span>
                        <span>{img.prediction.confidence.toFixed(2)}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5 mt-2">
                        <div
                          className={`h-2.5 rounded-full ${
                            img.prediction.label === "Covid"
                              ? "bg-red-600"
                              : img.prediction.label === "Viral Pneumonia"
                                ? "bg-amber-600"
                                : "bg-green-600"
                          }`}
                          style={{ width: `${img.prediction.confidence}%` }}
                        ></div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-2 text-gray-500">Ready for prediction</div>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

