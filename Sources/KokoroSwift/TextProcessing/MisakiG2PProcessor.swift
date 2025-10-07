#if canImport(MisakiSwift)

import Foundation
import MisakiSwift

class MisakiG2PProcessor : G2PProcessor {
  var misaki: EnglishG2P?
  
  func setLanguage(_ language: Language) throws {
    switch language {
    case .enUS:
      misaki = EnglishG2P(british: false)
    case .enGB:
      misaki = EnglishG2P(british: true)
    default:
      throw G2PProcessorError.unsupportedLanguage
    }
  }
  
  func process(input: String) throws -> String {
    guard let misaki else { throw G2PProcessorError.processorNotInitialized }
    return misaki.phonemize(text: input).0
  }

  func processWithTokens(input: String) throws -> (phonemes: String, tokens: [MisakiSwift.MToken]) {
    guard let misaki else { throw G2PProcessorError.processorNotInitialized }
    let result = misaki.phonemize(text: input)
    return (phonemes: result.0, tokens: result.1)
  }
}

#endif
