import { DiagnosisBadge } from "./diagnosis-badge"

interface Prediction {
  label: string
  confidence: number
}

interface StatsSummaryProps {
  predictions: Prediction[]
}

export function StatsSummary({ predictions }: StatsSummaryProps) {
  if (!predictions || predictions.length === 0) {
    return null
  }

  // Count occurrences of each label
  const counts = predictions.reduce(
    (acc, { label }) => {
      acc[label] = (acc[label] || 0) + 1
      return acc
    },
    {} as Record<string, number>,
  )

  // Calculate percentages
  const total = predictions.length
  const percentages = Object.entries(counts).map(([label, count]) => ({
    label,
    percentage: Math.round((count / total) * 100),
    count,
  }))

  return (
    <div className="bg-white rounded-lg shadow p-4 mb-8">
      <h2 className="text-lg font-semibold mb-4">Diagnosis Summary</h2>
      <div className="space-y-4">
        {percentages.map(({ label, percentage, count }) => (
          <div key={label} className="space-y-1">
            <div className="flex justify-between items-center">
              <div className="flex items-center gap-2">
                <DiagnosisBadge label={label} />
                <span className="text-sm font-medium">{count} images</span>
              </div>
              <span className="text-sm font-medium">{percentage}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className={`h-2 rounded-full ${
                  label === "Covid" ? "bg-red-600" : label === "Viral Pneumonia" ? "bg-amber-600" : "bg-green-600"
                }`}
                style={{ width: `${percentage}%` }}
              ></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

